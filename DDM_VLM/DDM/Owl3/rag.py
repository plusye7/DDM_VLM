import os
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
import clip
from torchvision import transforms
from scipy import ndimage
import string
from collections import Counter
import pandas as pd

from DDM.config import config_get, load_model_config
# Model import
from DDM.models.Owl3.Owl3 import OWL3

CONFIG = load_model_config("Owl3")

# Hyperparameters and paths
TOPK = int(config_get(CONFIG, "rag.topk", 10))

TEST_PATH = config_get(CONFIG, "data.test_path")

DATABASE_PATH = config_get(CONFIG, "data.database_path")
INDEX_PATH = config_get(CONFIG, "retrieval.index_path")

OUTPUT_DIR = config_get(CONFIG, "outputs.rag_dir")

# Class definitions
defects = [
    "center",
    "donut",
    "edge_loc",
    "edge_ring",
    "loc",
    "near_full",
    "scratch",
    "random",
]

defects_description = {
    "center": "Defects are mainly concentrated near the center of the image.",
    "donut": "A large defect is mainly concentrated near the center of the image, but the defect block is empty inside.",
    "edge_loc": "The main defects are concentrated in the edge area of the image, forming an arc shape, but not covering the entire edge.",
    "edge_ring": "The main defects are concentrated in the edge area of the image, forming a ring shape, covering the entire edge.",
    "loc": "Defects are mainly concentrated near the periphery of the image.",
    "near_full": "Defects almost cover the entire wafer area, exceeding 3/4 of the total area.",
    "scratch": "Long, linear defects resembling scratches exist in the image.",
    "random": "Defects are randomly distributed throughout the image, appearing in various locations, but their area does not fully cover the entire image; the total area does not exceed 3/4.",
}

# Load CLIP once at startup.
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP on {device} ...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load retrieval resources once at startup.
FAISS_INDEX = None
DATABASE_MAPS = None
DATABASE_LABELS = None

def initialize_retrieval_resources():
    """
    Load the FAISS index and retrieval database once at startup,
    then reuse the in-memory resources for all test samples.
    """
    global FAISS_INDEX, DATABASE_MAPS, DATABASE_LABELS

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file does not exist: {INDEX_PATH}")

    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(f"Retrieval database does not exist: {DATABASE_PATH}")

    print("Loading FAISS index ...")
    FAISS_INDEX = faiss.read_index(INDEX_PATH)

    print("Loading retrieval database ...")
    database_data = np.load(DATABASE_PATH)
    DATABASE_MAPS = database_data["arr_0"]
    DATABASE_LABELS = database_data["arr_1"]

    print(
        f"Retrieval resources loaded: "
        f"{len(DATABASE_MAPS)} database samples, "
        f"{FAISS_INDEX.ntotal} FAISS vectors."
    )

# Feature extraction
def extract_feature(image, model_type="clip"):
    if model_type == "clip":
        img = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(img)

        vector = feat.cpu().numpy()[0]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # L2-normalize for FAISS cosine retrieval.
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.astype(np.float32)

# Wafer map preprocessing
def process_wafer_map(wafer, min_cluster_size=10):
    """
    Filter small isolated defect clusters.
    wafer: raw wafer data (52x52)
    min_cluster_size: minimum connected-component size; smaller regions are removed
    """
    defect_mask = (wafer == 2).astype(np.uint8)

    structure = np.ones((3, 3), dtype=np.int8)
    labeled_mask, num_features = ndimage.label(
        defect_mask,
        structure=structure
    )

    component_sizes = ndimage.sum(
        defect_mask,
        labeled_mask,
        range(1, num_features + 1)
    )

    filtered_mask = np.zeros_like(defect_mask, dtype=np.uint8)

    for i, size in enumerate(component_sizes, start=1):
        if size >= min_cluster_size:
            filtered_mask[labeled_mask == i] = 1

    wafer_color = np.zeros((52, 52, 3), dtype=np.uint8)

    wafer_color[(wafer == 0) | (wafer == 1)] = [0, 0, 0]
    wafer_color[filtered_mask == 1] = [255, 0, 0]

    image = Image.fromarray(wafer_color)
    image = image.resize((384, 384), Image.Resampling.NEAREST)

    return image

# FAISS retrieval
def get_topk_similarities(processed_image):
    """
    Input is the target image after process_wafer_map preprocessing.
    The FAISS index is not reloaded per sample.
    """
    if FAISS_INDEX is None:
        raise RuntimeError(
            "FAISS index is not initialized. "
            "Call initialize_retrieval_resources() first."
        )

    vector = extract_feature(
        processed_image,
        model_type="clip"
    )

    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    vector = vector.astype(np.float32)

    faiss.normalize_L2(vector)

    similarities, indices = FAISS_INDEX.search(
        vector,
        TOPK
    )

    return indices[0], similarities[0]

def retrieve_topk(processed_image):
    """
    Retrieve Top-K items from the in-memory database.
    The database npz is not loaded again for each test sample.
    """
    if DATABASE_MAPS is None or DATABASE_LABELS is None:
        raise RuntimeError(
            "Retrieval database is not initialized. "
            "Call initialize_retrieval_resources() first."
        )

    indices, similarities = get_topk_similarities(
        processed_image
    )

    images_list = np.array([
        process_wafer_map(DATABASE_MAPS[i])
        for i in indices
    ])

    labels_list = DATABASE_LABELS[indices]

    return images_list, labels_list, similarities

# RAG example text
def build_example_texts(
    example_labels,
    similarities,
    label_dict
):
    """
    Each example tells the model:
    1. Original one-hot label
    2. Included defect types
    3. Similarity to the query image
    4. Candidate option letter
    """
    example_texts = []

    label_to_letter = {
        tuple(v): k
        for k, v in label_dict.items()
    }

    for label, sim in zip(
        example_labels,
        similarities
    ):
        onehot_str = list(label)

        defect_names = [
            defects[i]
            for i, v in enumerate(label)
            if v == 1
        ]

        if len(defect_names) == 0:
            defect_text = "no defect"
        else:
            defect_text = ", ".join(defect_names)

        letter = label_to_letter.get(
            tuple(label),
            "?"
        )

        text = (
            f"The one-hot label of this example is {onehot_str},\n "
            f"and it contains the following defect types: {defect_text}. \n"
            f"The similarity between this example image and the subsequent "
            f"image to be classified is {sim:.4f}.\n"
            f"The correct answer for this example is {letter}.\n"
        )

        example_texts.append(text)

    return example_texts

def build_label_mapping(labels_list):
    """
    Deduplicate labels that appear in the Top-K results,
    then map them to a / b / c / ...
    """
    unique_labels = []
    seen = set()

    for label in labels_list:
        label_tuple = tuple(label)

        if label_tuple not in seen:
            seen.add(label_tuple)
            unique_labels.append(
                np.array(label)
            )

    letters = string.ascii_lowercase
    label_dict = {}

    for i, label in enumerate(unique_labels):
        key = letters[i]
        label_dict[key] = label.tolist()

    print(label_dict)

    return label_dict

def compute_margin(probs):
    if probs.size < 2:
        return 1

    sorted_probs = np.sort(probs)

    return (
        sorted_probs[-1]
        - sorted_probs[-2]
    )

# Extract candidate label logits
def extract_class_logits(
    result,
    tokenizer,
    label_dict
):
    class_labels = list(
        label_dict.keys()
    )

    logits = result["generation_logits"]
    generated_ids = result[
        "generated_token_ids"
    ]

    label_token_ids = {}

    for letter in class_labels:
        ids = tokenizer.encode(
            letter,
            add_special_tokens=False
        )

        if len(ids) != 1:
            raise ValueError(
                f"{letter} is not single token: {ids}"
            )

        label_token_ids[letter] = ids[0]

    # Find the final classification-letter token from the end.
    class_step = None

    for i in range(
        len(generated_ids) - 1,
        -1,
        -1
    ):
        token_id = int(
            generated_ids[i]
        )

        for letter, lid in label_token_ids.items():
            if token_id == lid:
                class_step = i
                break

        if class_step is not None:
            break

    if class_step is None:
        print("❌ No class token found")
        print(
            "Generated:",
            tokenizer.decode(generated_ids)
        )
        return None

    # logits[t] predicts token[t+1].
    logit_step = max(
        class_step - 1,
        0
    )

    step_logits = logits[logit_step]

    class_logits = []

    for letter in class_labels:
        token_id = label_token_ids[
            letter
        ]

        class_logits.append(
            step_logits[token_id]
        )

    return np.array(
        class_logits,
        dtype=np.float32
    )

# Prompt
def create_prompt(label_dict):
    label_lines = []

    for letter, label in label_dict.items():
        label_lines.append(
            f"{letter}: {label}"
        )

    label_text = "\n".join(
        label_lines
    )

    return f"""
    Role: You are a professional semiconductor yield engineer.

    I will give you a wafer image, where black represents normal areas and red represents defective areas.
    I will also give you {TOPK} examples, which are the examples with the highest similarity retrieved from the target image. The correct answer is the same as one of these examples.
    Your task is to analyze the shape and location of the connected blocks of defect points (red pixels) in both the target image and the example images. Analyze whether each type of defect in the example image exists in the target image, and determine the defect type in the target image based on this.

    The following is an explanation of the defect types in the images:
    - normal: No red pixels.
    - Center: Defects are mainly concentrated near the center of the image.
    - Donut: A large defect is mainly concentrated near the center of the image, but the defect block is empty inside.
    - Edge_Loc: The main defects are concentrated in the edge area of the image, forming an arc shape, but not covering the entire edge.
    - Edge_Ring: The main defects are concentrated in the edge area of the image, forming a ring shape, covering the entire edge.
    - Loc: Defects are mainly concentrated near the periphery of the image.
    - Near_Full: Defects almost cover the entire wafer area, exceeding 3/4 of the total area.
    - Scratch: Long, linear defects resembling scratches exist in the image.
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

    Final answer format:
    <One letter only>
    """

def majority_vote_onehot(labels_onehot):
    label_tuples = [
        tuple(label)
        for label in labels_onehot
    ]

    counter = Counter(
        label_tuples
    )

    majority_tuple = (
        counter.most_common(1)[0][0]
    )

    return np.array(
        majority_tuple
    )

# Single-sample prediction
def predict(image):
    # Preprocess the target wafer once.
    # Reuse the processed image for both CLIP retrieval and OWL3.
    image_input = process_wafer_map(
        image
    )

    images_list, labels_list, similarities = retrieve_topk(
        image_input
    )

    label_dict = build_label_mapping(
        labels_list
    )

    example_texts = build_example_texts(
        labels_list,
        similarities,
        label_dict
    )

    prompt_text = create_prompt(
        label_dict
    )

    result = OWL3.predict(
        prompt_text,
        image_input,
        example_texts=example_texts,
        example_images=images_list
    )

    class_logits = extract_class_logits(
        result,
        OWL3._tokenizer,
        label_dict
    )

    if class_logits is None:
        raise RuntimeError(
            "Failed to extract class logits."
        )

    probs = torch.softmax(
        torch.tensor(class_logits),
        dim=-1
    ).numpy().astype(
        "float32"
    )

    probs = probs.flatten()

    pred_idx = np.argmax(
        probs
    )

    letters = list(
        label_dict.keys()
    )

    pred_letter = letters[
        pred_idx
    ]

    pred_onehot = np.array(
        label_dict[pred_letter]
    )

    return pred_onehot, probs

# Metrics
def compute_metrics(
    preds,
    labels
):
    exact = np.all(
        preds == labels,
        axis=1
    )

    hamming = (
        preds == labels
    )

    per_class_f1 = {}
    f1_values = []

    for idx, name in enumerate(defects):
        pred = preds[:, idx] == 1
        true = labels[:, idx] == 1

        tp = np.logical_and(
            pred,
            true
        ).sum()

        fp = np.logical_and(
            pred,
            ~true
        ).sum()

        fn = np.logical_and(
            ~pred,
            true
        ).sum()

        denom = (
            2 * tp
            + fp
            + fn
        )

        if denom == 0:
            f1 = 0.0
        else:
            f1 = float(
                (2 * tp) / denom
            )

        per_class_f1[
            name
        ] = f1

        f1_values.append(
            f1
        )

    correct_count = int(
        exact.sum()
    )

    num_samples = int(
        len(labels)
    )

    return {
        "exact_match_accuracy": float(
            exact.mean()
        ),
        "correct_count": correct_count,
        "incorrect_count": (
            num_samples
            - correct_count
        ),
        "hamming_accuracy": float(
            hamming.mean()
        ),
        "macro_f1": float(
            np.mean(f1_values)
        ),
        "per_class_f1": per_class_f1,
    }

# Main function
def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    result_csv = os.path.join(
        OUTPUT_DIR,
        "rag_result.csv"
    )

    probs_csv = os.path.join(
        OUTPUT_DIR,
        "probs.csv"
    )

    summary_json = os.path.join(
        OUTPUT_DIR,
        "summary.json"
    )

    # Load the test set once.
    print("Loading test dataset ...")

    test_data = np.load(
        TEST_PATH
    )

    test_maps = test_data[
        "arr_0"
    ]

    test_labels = test_data[
        "arr_1"
    ]

    all_predictions = []
    all_probs = []

    # Predict sample by sample.
    for idx in tqdm(
        range(len(test_maps))
    ):
        image = test_maps[idx]

        onehot, probs = predict(
            image
        )

        all_predictions.append(
            onehot
        )

        all_probs.append(
            tuple(probs)
        )

        is_correct = np.array_equal(
            onehot,
            test_labels[idx]
        )

        print(
            f"Sample {idx}: {is_correct}"
        )

    all_predictions = np.asarray(
        all_predictions,
        dtype=np.int32
    )

    test_labels = np.asarray(
        test_labels,
        dtype=np.int32
    )

    # Save predictions.
    df = pd.DataFrame(
        all_predictions,
        columns=defects
    )

    df.to_csv(
        result_csv,
        index=False
    )

    # Save probabilities.
    # Each sample probability vector length depends on
    # the number of deduplicated candidate labels in Top-K,
    # so save it as a tuple.
    df_probs = pd.DataFrame({
        "probs": all_probs
    })

    df_probs.to_csv(
        probs_csv,
        index=False
    )

    # Compute metrics.
    metrics = compute_metrics(
        all_predictions,
        test_labels
    )

    # Valid joint-label count
    # DATABASE_LABELS is already in memory,
    # so the database is not loaded again here.
    valid_labels = np.unique(
        DATABASE_LABELS,
        axis=0
    )

    valid_label_count = int(
        len(valid_labels)
    )

    # summary.json
    summary = {
        "model": "OWL3",
        "method": "pure_rag",
        "test_path": TEST_PATH,
        "database_path": DATABASE_PATH,
        "index_path": INDEX_PATH,
        "num_samples": int(
            len(test_maps)
        ),
        "defects": defects,
        "valid_label_count": valid_label_count,
        "hyperparameters": {
            "topk": TOPK,
            "min_cluster_size": 10,
            "retrieval_model": "CLIP ViT-B/32",
        },
        "metrics": metrics,
        "avg_vlm_calls": 1.0,
        "output_files": {
            "predictions": result_csv,
            "probabilities": probs_csv,
            "summary": summary_json,
        },
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

    print(
        "\n================================"
    )
    print("✅ Finished.")
    print(
        "================================"
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        f"\nPredictions saved to: {result_csv}"
    )

    print(
        f"Probabilities saved to: {probs_csv}"
    )

    print(
        f"Summary saved to: {summary_json}"
    )

if __name__ == "__main__":
    # 1. Initialize OWL3 once.
    print("Initializing OWL3 ...")
    OWL3.initialize()

    # 2. Load FAISS and the database once.
    initialize_retrieval_resources()

    # 3. Start evaluation.
    main()
