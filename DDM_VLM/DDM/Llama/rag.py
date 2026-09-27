import os
import sys
import numpy as np
import torch
import faiss
from tqdm import tqdm
import json
from PIL import Image
import clip
from torchvision import models, transforms
from scipy import ndimage
import string
from collections import Counter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from DDM.config import config_get, load_model_config
# Model import
from DDM.models.Llama.llama import Llama

CONFIG = load_model_config("Llama")

# Hyperparameters
TOPK = int(config_get(CONFIG, "rag.topk", 10))

# Class definitions
defects=['center','donut','edge_loc','edge_ring','loc','near_full','scratch','random']

defects_description={
    'center': "Defects are mainly concentrated near the center of the image.",
    'donut': "A large defect is mainly concentrated near the center of the image, but the defect block is empty inside.",
    'edge_loc': "The main defects are concentrated in the edge area of ​​the image, forming an arc shape, but not covering the entire edge.",
    'edge_ring': "The main defects are concentrated in the edge area of ​​the image, forming a ring shape, covering the entire edge.",
    'loc': "Defects are mainly concentrated near the periphery of the image.",
    'near_full': "Defects almost cover the entire wafer area, exceeding 3/4 of the total area.",
    'scratch': "Long, linear defects resembling scratches exist in the image.",
    'random': "Defects are randomly distributed throughout the image, appearing in various locations, but their area does not fully cover the entire image; the total area does not exceed 3/4."
}

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
INDEX_PATH = config_get(CONFIG, "retrieval.index_path")
DATABASE_PATH = config_get(CONFIG, "data.database_path")

print("Loading FAISS index...")
faiss_index = faiss.read_index(INDEX_PATH)

print("Loading retrieval database...")
database_data = np.load(DATABASE_PATH)

database_maps = database_data["arr_0"]
database_labels = database_data["arr_1"]

print("Retrieval resources loaded.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_feature(image, model_type="clip"):
    if model_type in ["vgg", "resnet"]:
        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            if model_type == "vgg":
                feat = vgg_features(img)
                feat = vgg_avgpool(feat)

            elif model_type == "resnet":
                feat = resnet(img)

        feat = feat.view(feat.size(0), -1)
        vector = feat.cpu().numpy()[0]

    elif model_type == "clip":

        img = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(img)

        vector = feat.cpu().numpy()[0]

    # L2-normalize for FAISS cosine retrieval.
    vector = vector / np.linalg.norm(vector)

    return vector.astype(np.float32)

# Utility functions
def process_wafer_map(wafer, min_cluster_size=10):
    """
    Filter small isolated defect clusters.
    :param wafer: raw wafer data (52x52)
    :param min_cluster_size: minimum connected-component size; smaller regions are removed
    """
    # Create the defect mask; value 2 marks defects.
    defect_mask = (wafer == 2).astype(np.uint8)

    # Use 8-neighbor connected components for wafer defects.
    structure = np.ones((3, 3), dtype=np.int8)
    labeled_mask, num_features = ndimage.label(defect_mask, structure=structure)

    # Compute each connected component size.
    component_sizes = ndimage.sum(defect_mask, labeled_mask, range(1, num_features + 1))

    # Build the retained-component mask.
    filtered_mask = np.zeros_like(defect_mask, dtype=np.uint8)

    for i, size in enumerate(component_sizes, start=1):
        if size >= min_cluster_size:   # Keep only components with size >= 10.
            filtered_mask[labeled_mask == i] = 1

    # Build the color image.
    wafer_color = np.zeros((52, 52, 3), dtype=np.uint8)

    # Background and normal regions.
    wafer_color[(wafer == 0) | (wafer == 1)] = [0, 0, 0]

    # Retained defects.
    wafer_color[filtered_mask == 1] = [255, 0, 0]

    image = Image.fromarray(wafer_color)
    image = image.resize((384, 384), Image.Resampling.NEAREST)
    return image

def get_topk_similarities(processed_image):

    vector = extract_feature(
        processed_image,
        model_type="clip"
    )

    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    vector = vector.astype(np.float32)

    faiss.normalize_L2(vector)

    similarities, indices = faiss_index.search(
        vector,
        TOPK
    )

    return indices[0], similarities[0]

def retrieve_topk(processed_image):

    indices, similarities = get_topk_similarities(
        processed_image
    )

    images_list = np.array([
        process_wafer_map(database_maps[i])
        for i in indices
    ])

    labels_list = database_labels[indices]

    return images_list, labels_list, similarities

def build_example_texts(example_labels, similarities,label_dict):
    """
    Each example tells the model:
    Original one-hot label
    Included defect types
    Similarity to the query image
    """
    example_texts = []

    label_to_letter = {
        tuple(v): k for k, v in label_dict.items()
    }

    for label, sim in zip(example_labels, similarities):

        # Convert one-hot labels to lists for display.
        onehot_str = list(label)

        # Find active defects in the one-hot label.
        defect_names = [
            defects[i] for i, v in enumerate(label) if v == 1
        ]

        if len(defect_names) == 0:
            defect_text = "no defect"
        else:
            defect_text = ", ".join(defect_names)

        letter = label_to_letter.get(tuple(label), "?")

        text = (
            f"The one-hot label of this example is {onehot_str},\n "
            f"and it contains the following defect types: {defect_text}. \n"
            f"The similarity between this example image and the subsequent image to be classified is {sim:.4f}.\n"
            f"The correct answer for this example is {letter}.\n"
        )

        example_texts.append(text)

    return example_texts

def build_label_mapping(labels_list):
    """
    labels_list: Top-K labels returned by FAISS retrieval (list of one-hot arrays)

    return:
        label_dict: { 'a': label1, 'b': label2, ... }
    """

    # Deduplicate
    unique_labels = []
    seen = set()

    for label in labels_list:
        label_tuple = tuple(label)

        if label_tuple not in seen:
            seen.add(label_tuple)
            unique_labels.append(np.array(label))

    # Letter mapping
    letters = string.ascii_lowercase

    label_dict = {}

    for i, label in enumerate(unique_labels):
        key = letters[i]
        label_dict[key] = label.tolist()

    print(label_dict)
    return label_dict

def compute_margin(probs):
    if probs.size < 2 :
        return 1

    sorted_probs = np.sort(probs)
    return sorted_probs[-1] - sorted_probs[-2]

def extract_class_logits(result, tokenizer, label_dict):
    class_labels = list(label_dict.keys())

    logits = result["generation_logits"]
    generated_ids = result["generated_token_ids"]

    if len(generated_ids) < 2:
        print("❌ generated sequence too short")
        return None

    # Core logic: take logits from the penultimate token.
    # logits[t] predicts token[t+1].
    step_logits = logits[-2]

    # Extract logits for each class.
    class_logits = []

    for label in class_labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)

        if len(token_ids) != 1:
            raise ValueError(
                f"Label '{label}' is not single token under current tokenizer: {token_ids}"
            )

        class_logits.append(step_logits[token_ids[0]])

    return np.array(class_logits, dtype=np.float32)

def create_prompt(label_dict):
    """Build the prompt."""
    label_lines = []

    for letter, label in label_dict.items():
        label_lines.append(f"{letter}: {label}")

    label_text = "\n".join(label_lines)

    return f"""
    Role: You are a professional semiconductor yield engineer.

    I will give you a wafer image, where black represents normal areas and red represents defective areas.
    I will also give you {TOPK} examples, which are the examples with the highest similarity retrieved from the target image. The correct answer is the same as one of these examples.
    Your task is to analyze the shape and location of the connected blocks of defect points (red pixels) in both the target image and the example images. Analyze whether each type of defect in the example image exists in the target image, and determine the defect type in the target image based on this.

    The following is an explanation of the defect types in the images:
    - normal: No red pixels.
    - Center: Defects are mainly concentrated near the center of the image.
    - Donut: A large defect is mainly concentrated near the center of the image, but the defect block is empty inside.
    - Edge_Loc: The main defects are concentrated in the edge area of ​​the image, forming an arc shape, but not covering the entire edge.
    - Edge_Ring: The main defects are concentrated in the edge area of ​​the image, forming a ring shape, covering the entire edge.
    - Loc: Defects are mainly concentrated near the periphery of the image.
    - Near_Full: Defects almost cover the entire wafer area, exceeding 3/4 of the total area.
    - Scratch:Long, linear defects resembling scratches exist in the image.
    - Random: Defects are randomly distributed throughout the image, appearing in various locations, but their area does not fully cover the entire image; the total area does not exceed 3/4.

    Important constraints:
    - The wafer must either be normal or have the same label as a given example.
    - Focus on comparing the defect blocks in each of the target and example images, analyzing the presence of each type of defect.
    - Pay primary attention to the location of defect clusters, as well as their shapes and coverage areas.
    - Select only one category from the following list.
    - Output only one letter.

    Where:
    Label options (letter -> one-hot label):
    {label_text}

    ⚠️ STRICT OUTPUT RULES (VERY IMPORTANT):
    - You MUST output exactly ONE letter from the options above.
    - Do NOT output any explanation.
    - Do NOT output words, sentences, or punctuation.
    - Do NOT output anything except a single character.
    - If unsure, still choose the most likely letter.

    Final answer format:
    <One letter only>

    """

def majority_vote_onehot(labels_onehot):
    label_tuples = [tuple(label) for label in labels_onehot]
    counter = Counter(label_tuples)
    majority_tuple = counter.most_common(1)[0][0]
    majority_onehot = np.array(majority_tuple)
    return majority_onehot

def predict(image):

    # Process the target once.
    image_input = process_wafer_map(image)

    images_list, labels_list, similarities = retrieve_topk(
        image_input
    )

    label_dict = build_label_mapping(labels_list)

    example_texts = build_example_texts(
        labels_list,
        similarities,
        label_dict
    )

    prompt_text = create_prompt(label_dict)

    result = Llama.predict(
        prompt_text,
        image_input,
        example_texts=example_texts,
        example_images=images_list
    )

    class_logits = extract_class_logits(
        result,
        Llama._tokenizer,
        label_dict
    )

    probs = torch.softmax(
        torch.tensor(class_logits),
        dim=-1
    ).numpy().astype("float32")

    probs = probs.flatten()

    pred_idx = np.argmax(probs)

    letters = list(label_dict.keys())
    pred_letter = letters[pred_idx]

    pred_onehot = np.array(
        label_dict[pred_letter]
    )

    return pred_onehot, probs

def compute_metrics(preds, labels):
    exact = np.all(preds == labels, axis=1)
    hamming = preds == labels

    per_class_f1 = {}
    f1_values = []

    for idx, name in enumerate(defects):
        pred = preds[:, idx] == 1
        true = labels[:, idx] == 1

        tp = np.logical_and(pred, true).sum()
        fp = np.logical_and(pred, ~true).sum()
        fn = np.logical_and(~pred, true).sum()

        denom = 2 * tp + fp + fn

        f1 = (
            0.0
            if denom == 0
            else float((2 * tp) / denom)
        )

        per_class_f1[name] = f1
        f1_values.append(f1)

    return {
        "exact_match_accuracy": float(exact.mean()),
        "hamming_accuracy": float(hamming.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class_f1": per_class_f1,
    }

def main():
    test_path = config_get(CONFIG, "data.test_path")
    database_path = config_get(CONFIG, "data.database_path")
    index_path = config_get(CONFIG, "retrieval.index_path")
    output_dir = config_get(CONFIG, "outputs.rag_dir")

    os.makedirs(output_dir, exist_ok=True)

    result_csv = os.path.join(
        output_dir,
        "rag_result.csv"
    )

    probs_csv = os.path.join(
        output_dir,
        "probs.csv"
    )

    summary_json = os.path.join(
        output_dir,
        "summary.json"
    )

    test_data = np.load(test_path)

    test_maps = test_data["arr_0"]
    test_labels = test_data["arr_1"]

    all_predictions = []
    all_probs = []

    for idx in tqdm(range(len(test_maps))):

        image = test_maps[idx]

        onehot, probs = predict(image)

        all_predictions.append(onehot)
        all_probs.append(tuple(probs))

        is_correct = np.array_equal(
            onehot,
            test_labels[idx]
        )

        print(
            f"Sample {idx}: {is_correct} \n\n\n"
        )

    all_predictions = np.asarray(
        all_predictions,
        dtype=np.int32
    )

    test_labels = np.asarray(
        test_labels,
        dtype=np.int32
    )

    df = pd.DataFrame(all_predictions)

    df.to_csv(
        result_csv,
        index=False
    )

    df_probs = pd.DataFrame({
        "probs": all_probs
    })

    df_probs.to_csv(
        probs_csv,
        index=False
    )

    database_data = np.load(database_path)
    database_labels = database_data["arr_1"]

    valid_labels = np.unique(
        database_labels,
        axis=0
    )

    valid_label_count = len(valid_labels)

    metrics = compute_metrics(
        all_predictions,
        test_labels
    )

    summary = {

        "test_path": test_path,

        "database_path": database_path,

        "index_path": index_path,

        "num_samples": int(len(test_maps)),

        "defects": defects,

        "valid_label_count": int(
            valid_label_count
        ),

        "hyperparameters": {
            "topk": TOPK
        },

        "metrics": metrics,

        "avg_vlm_calls": 1.0,

        "result_csv": result_csv,

        "probs_csv": probs_csv
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n================================")
    print("✅ Finished.")
    print("================================")

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        f"\nResult saved to: {result_csv}"
    )

    print(
        f"Probabilities saved to: {probs_csv}"
    )

    print(
        f"Summary saved to: {summary_json}"
    )

if __name__ == "__main__":
    Llama.initialize()
    main()
