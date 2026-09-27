import argparse
import csv
import json
import math
import os
import string
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from DDM.config import config_get, load_config_from_args
from DDM.models.InternVL3.InternVL3 import InternVL3

DEFECTS = [
    "center",
    "donut",
    "edge_loc",
    "edge_ring",
    "loc",
    "near_full",
    "scratch",
    "random",
]

DEFECT_DESCRIPTIONS = {
    "center": "Defects are mainly concentrated near the center of the image.",
    "donut": (
        "A large defect is mainly concentrated near the center of the image, "
        "but the defect block is empty inside."
    ),
    "edge_loc": (
        "The main defects are concentrated in the edge area of the image, "
        "forming an arc shape, but not covering the entire edge."
    ),
    "edge_ring": (
        "The main defects are concentrated in the edge area of the image, "
        "forming a ring shape, covering the entire edge."
    ),
    "loc": "Defects are mainly concentrated near the periphery of the image.",
    "near_full": (
        "Defects almost cover the entire wafer area, exceeding 3/4 of the "
        "total area."
    ),
    "scratch": "Long, linear defects resembling scratches exist in the image.",
    "random": (
        "Defects are randomly distributed throughout the image, appearing in "
        "various locations, but their area does not fully cover the entire "
        "image; the total area does not exceed 3/4."
    ),
}

CLIP_MODEL = None
CLIP_PREPROCESS = None
FAISS_MODULE = None
TORCH_MODULE = None
DEVICE = None

def ensure_torch():
    global TORCH_MODULE
    if TORCH_MODULE is None:
        import torch

        TORCH_MODULE = torch
    return TORCH_MODULE

def get_device():
    global DEVICE
    if DEVICE is None:
        torch = ensure_torch()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE

def ensure_clip():
    global CLIP_MODEL, CLIP_PREPROCESS
    if CLIP_MODEL is None or CLIP_PREPROCESS is None:
        import clip

        CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=get_device())
    return CLIP_MODEL, CLIP_PREPROCESS

def ensure_faiss():
    global FAISS_MODULE
    if FAISS_MODULE is None:
        import faiss

        FAISS_MODULE = faiss
    return FAISS_MODULE

def as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value

def parse_split(raw):
    if isinstance(raw, list):
        split = [int(x) for x in raw]
    else:
        split = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not split or any(x <= 0 for x in split):
        raise ValueError("--retrieval-split must contain positive integers.")
    return split

def process_wafer_map(wafer, min_cluster_size=10):
    defect_mask = (wafer == 2).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.int8)
    labeled_mask, num_features = ndimage.label(defect_mask, structure=structure)
    component_sizes = ndimage.sum(
        defect_mask,
        labeled_mask,
        range(1, num_features + 1),
    )

    filtered_mask = np.zeros_like(defect_mask, dtype=np.uint8)
    for idx, size in enumerate(component_sizes, start=1):
        if size >= min_cluster_size:
            filtered_mask[labeled_mask == idx] = 1

    wafer_color = np.zeros((52, 52, 3), dtype=np.uint8)
    wafer_color[(wafer == 0) | (wafer == 1)] = [0, 0, 0]
    wafer_color[filtered_mask == 1] = [255, 0, 0]

    image = Image.fromarray(wafer_color)
    return image.resize((384, 384), Image.Resampling.NEAREST)

def extract_feature(image):
    torch = ensure_torch()
    clip_model, clip_preprocess = ensure_clip()
    img = clip_preprocess(image).unsqueeze(0).to(get_device())
    with torch.no_grad():
        feature = clip_model.encode_image(img)

    vector = feature.cpu().numpy()[0]
    vector = vector / np.linalg.norm(vector)
    return vector.astype(np.float32)

def build_valid_label_space(labels):
    counts = Counter(tuple(int(v) for v in row) for row in labels)
    valid_labels = np.array(sorted(counts.keys()), dtype=np.int32)
    valid_label_index = {
        tuple(int(v) for v in label): idx
        for idx, label in enumerate(valid_labels)
    }
    priors = np.array(
        [math.log(counts[tuple(label)] / len(labels)) for label in valid_labels],
        dtype=np.float32,
    )
    return valid_labels, valid_label_index, priors

def load_resources(args):
    database = np.load(args.database_path)
    database_maps = database["arr_0"]
    database_labels = database["arr_1"].astype(np.int32)

    valid_labels, valid_label_index, label_log_priors = build_valid_label_space(
        database_labels
    )

    faiss = ensure_faiss()
    return {
        "index": faiss.read_index(args.index_path),
        "database_maps": database_maps,
        "database_labels": database_labels,
        "valid_labels": valid_labels,
        "valid_label_index": valid_label_index,
        "label_log_priors": label_log_priors,
    }

def retrieve_topk(image, resources, topk):
    faiss = ensure_faiss()
    image_input = process_wafer_map(image)
    vector = extract_feature(image_input).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vector)

    similarities, indices = resources["index"].search(vector, topk)
    similarities = similarities[0]
    indices = indices[0]

    maps = resources["database_maps"]
    labels = resources["database_labels"]
    images = [process_wafer_map(maps[i]) for i in indices]
    retrieved_labels = labels[indices]

    return image_input, images, retrieved_labels, similarities

def label_to_text(label):
    names = [DEFECTS[i] for i, v in enumerate(label) if int(v) == 1]
    if not names:
        return "normal"
    return ", ".join(names)

def build_label_mapping(labels):
    unique_labels = []
    seen = set()
    for label in labels:
        key = tuple(int(v) for v in label)
        if key not in seen:
            seen.add(key)
            unique_labels.append(key)

    if len(unique_labels) > len(string.ascii_lowercase):
        raise ValueError("Too many candidate labels for single-letter decoding.")

    return {
        string.ascii_lowercase[idx]: list(label)
        for idx, label in enumerate(unique_labels)
    }

def build_example_texts(example_labels, similarities, label_dict):
    label_to_letter = {tuple(v): key for key, v in label_dict.items()}
    texts = []
    for label, sim in zip(example_labels, similarities):
        label_key = tuple(int(v) for v in label)
        letter = label_to_letter.get(label_key, "?")
        texts.append(
            "This retrieved wafer example has one-hot label "
            f"{list(map(int, label))}; defect types: {label_to_text(label)}. "
            f"Its candidate option is {letter}. "
            f"The similarity to the target wafer is {float(sim):.4f}."
        )
    return texts

def create_prompt(label_dict, round_id, max_rounds):
    label_lines = []
    for letter, label in label_dict.items():
        label_lines.append(f"{letter}: {label} ({label_to_text(label)})")
    label_text = "\n".join(label_lines)

    description_lines = []
    for name in DEFECTS:
        description_lines.append(f"- {name}: {DEFECT_DESCRIPTIONS[name]}")
    description_text = "\n".join(description_lines)

    return f"""
Role: You are a professional semiconductor yield engineer.

This is round {round_id} of {max_rounds}. I will give you a target wafer image
and several retrieved wafer examples. Black pixels are normal areas and red
pixels are defective areas.

Your task is to choose the most likely full joint defect label for the target
wafer. The final answer must be exactly one letter from the candidate label
options below.

Defect descriptions:
{description_text}

Candidate label options:
{label_text}

Important constraints:
- Compare the target wafer with the retrieved examples using defect location,
  shape, distribution, and coverage.
- Keep the full one-hot label combination intact. Do not decide each defect
  independently.
- Output only one letter.
- Do not output any explanation, punctuation, or other words.

Final answer format:
<One letter only>
"""

def output_fallback_logits(raw_output, class_labels):
    raw = str(raw_output).strip().lower()
    for char in raw:
        if char in class_labels:
            logits = np.full(len(class_labels), -20.0, dtype=np.float32)
            logits[class_labels.index(char)] = 0.0
            return logits
    return None

def class_token_ids(tokenizer, letter):
    candidates = [letter, f" {letter}", f"\n{letter}"]
    ids = []

    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1:
            ids.append(int(token_ids[0]))

    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError(
            f"Label '{letter}' has no single-token form under current tokenizer."
        )
    return ids

def extract_class_logits(result, tokenizer, label_dict):
    class_labels = list(label_dict.keys())
    logits = np.asarray(result.get("generation_logits", []))

    if logits.size == 0:
        fallback = output_fallback_logits(result.get("raw_output", ""), class_labels)
        if fallback is not None:
            return fallback, "raw_output_fallback"
        return None, "missing"

    generated_ids = np.asarray(result.get("generated_token_ids", []), dtype=np.int64)
    step_idx = None
    generated_texts = []

    for idx in range(len(generated_ids) - 1, -1, -1):
        token_text = tokenizer.decode([int(generated_ids[idx])]).strip().lower()
        generated_texts.append(token_text)
        if token_text in class_labels and idx < len(logits):
            step_idx = idx
            break

    if step_idx is None:
        fallback = output_fallback_logits(result.get("raw_output", ""), class_labels)
        if fallback is not None:
            return fallback, "raw_output_fallback"
        step_idx = 0
        prob_source = "vlm_first_step_logits"
    else:
        prob_source = "vlm_generation_logits"

    step_logits = logits[step_idx]
    class_logits = []
    for letter in class_labels:
        ids = class_token_ids(tokenizer, letter)
        class_logits.append(float(np.max(step_logits[ids])))

    finite = [x for x in class_logits if np.isfinite(x)]
    if not finite:
        fallback = output_fallback_logits(result.get("raw_output", ""), class_labels)
        if fallback is not None:
            return fallback, "raw_output_fallback"
        return None, "missing"

    floor = min(finite) - 20.0
    class_logits = np.array(
        [floor if not np.isfinite(x) else x for x in class_logits],
        dtype=np.float32,
    )
    return class_logits, "vlm_generation_logits"

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    values = np.exp(logits)
    return (values / values.sum()).astype(np.float32)

def vlm_label_evidence(candidate_probs, label_dict, resources, evidence_clip):
    valid_label_index = resources["valid_label_index"]
    evidence = np.zeros(len(resources["valid_labels"]), dtype=np.float32)
    if not candidate_probs:
        return evidence

    uniform = 1.0 / len(candidate_probs)
    for letter, prob in candidate_probs.items():
        label = tuple(int(v) for v in label_dict[letter])
        valid_idx = valid_label_index.get(label)
        if valid_idx is None:
            continue

        value = math.log((float(prob) + 1e-8) / uniform)
        evidence[valid_idx] += value

    return np.clip(evidence, -evidence_clip, evidence_clip).astype(np.float32)

def retrieval_label_evidence(labels, similarities, resources, tau, evidence_clip):
    valid_label_index = resources["valid_label_index"]
    n_valid = len(resources["valid_labels"])
    probs = np.full(n_valid, 1e-8, dtype=np.float64)

    labels = np.asarray(labels, dtype=np.int32)
    similarities = np.asarray(similarities, dtype=np.float64)
    weights = np.exp((similarities - similarities.max()) / tau)

    for label, weight in zip(labels, weights):
        key = tuple(int(v) for v in label)
        valid_idx = valid_label_index.get(key)
        if valid_idx is not None:
            probs[valid_idx] += float(weight)

    probs /= probs.sum()
    uniform = 1.0 / n_valid
    evidence = np.log(probs / uniform)
    return np.clip(evidence, -evidence_clip, evidence_clip).astype(np.float32)

def top_two(label_evidence, valid_labels):
    order = np.argsort(label_evidence)[::-1]
    best_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else best_idx
    margin = float(label_evidence[best_idx] - label_evidence[second_idx])
    return {
        "best_idx": best_idx,
        "second_idx": second_idx,
        "margin": margin,
        "best_label": valid_labels[best_idx].astype(int).tolist(),
        "second_label": valid_labels[second_idx].astype(int).tolist(),
    }

def format_label_scores(label_evidence, valid_labels, topn=5):
    order = np.argsort(label_evidence)[::-1][:topn]
    return {
        str(idx): {
            "score": float(label_evidence[idx]),
            "label": valid_labels[idx].astype(int).tolist(),
        }
        for idx in order
    }

def split_round_items(images, labels, similarities, split, round_idx, mode):
    if mode == "cumulative":
        start = 0
        end = sum(split[: round_idx + 1])
    else:
        start = sum(split[:round_idx])
        end = sum(split[: round_idx + 1])
    return images[start:end], labels[start:end], similarities[start:end]

def predict_one(image, resources, args, rng, sample_index=None):
    retrieval_topk = sum(args.retrieval_split)
    image_input, images, labels, similarities = retrieve_topk(
        image,
        resources,
        topk=retrieval_topk,
    )

    label_dict = build_label_mapping(labels)
    valid_labels = resources["valid_labels"]
    label_evidence = np.zeros(len(valid_labels), dtype=np.float32)

    trace_rows = []
    rounds_used = 0

    InternVL3.reset_chat()

    for round_idx in range(args.max_rounds):
        if round_idx >= len(args.retrieval_split):
            break

        round_images, round_labels, round_sims = split_round_items(
            images,
            labels,
            similarities,
            args.retrieval_split,
            round_idx,
            args.example_mode,
        )

        prompt = create_prompt(
            label_dict,
            round_id=round_idx + 1,
            max_rounds=args.max_rounds,
        )
        example_texts = build_example_texts(round_labels, round_sims, label_dict)

        result = InternVL3.predict(
            prompt,
            image_input,
            example_texts=example_texts,
            example_images=round_images,
            max_new_tokens=args.max_new_tokens,
        )

        logits, prob_source = extract_class_logits(
            result,
            InternVL3._tokenizer,
            label_dict,
        )

        if logits is None:
            probs = np.full(len(label_dict), 1.0 / len(label_dict), dtype=np.float32)
        else:
            probs = softmax(logits)

        letters = list(label_dict.keys())
        candidate_probs = {
            letter: float(prob)
            for letter, prob in zip(letters, probs)
        }

        vlm_evidence = vlm_label_evidence(
            candidate_probs,
            label_dict,
            resources,
            args.evidence_clip,
        )
        rag_evidence = retrieval_label_evidence(
            round_labels,
            round_sims,
            resources,
            args.tau,
            args.evidence_clip,
        )
        noise = rng.normal(
            0.0,
            args.sigma,
            size=len(valid_labels),
        ).astype(np.float32)

        delta = args.alpha * vlm_evidence + args.beta * rag_evidence + noise
        label_evidence += delta
        rounds_used += 1

        rank_info = top_two(label_evidence, valid_labels)
        stopped = args.early_stop and rank_info["margin"] >= args.boundary

        trace_rows.append(
            {
                "index": sample_index,
                "round": round_idx + 1,
                "retrieval_ranks": (
                    f"1-{sum(args.retrieval_split[: round_idx + 1])}"
                    if args.example_mode == "cumulative"
                    else (
                        f"{sum(args.retrieval_split[:round_idx]) + 1}-"
                        f"{sum(args.retrieval_split[: round_idx + 1])}"
                    )
                ),
                "prob_source": prob_source,
                "raw_output": result.get("raw_output", ""),
                "candidate_probs": json.dumps(as_jsonable(candidate_probs)),
                "top_candidate_labels": json.dumps(as_jsonable(label_dict)),
                "vlm_top_evidence": json.dumps(
                    as_jsonable(format_label_scores(vlm_evidence, valid_labels))
                ),
                "rag_top_evidence": json.dumps(
                    as_jsonable(format_label_scores(rag_evidence, valid_labels))
                ),
                "noise_top_values": json.dumps(
                    as_jsonable(format_label_scores(noise, valid_labels))
                ),
                "accumulated_top_evidence": json.dumps(
                    as_jsonable(format_label_scores(label_evidence, valid_labels))
                ),
                "best_label": json.dumps(as_jsonable(rank_info["best_label"])),
                "second_label": json.dumps(as_jsonable(rank_info["second_label"])),
                "margin": rank_info["margin"],
                "stopped": stopped,
            }
        )

        if stopped:
            break

    final_info = top_two(label_evidence, valid_labels)
    prediction = valid_labels[final_info["best_idx"]].astype(np.int32)
    return prediction, trace_rows, rounds_used

def compute_metrics(preds, labels):
    exact = np.all(preds == labels, axis=1)
    hamming = preds == labels

    per_class_f1 = {}
    f1_values = []
    for idx, name in enumerate(DEFECTS):
        pred = preds[:, idx] == 1
        true = labels[:, idx] == 1
        tp = np.logical_and(pred, true).sum()
        fp = np.logical_and(pred, ~true).sum()
        fn = np.logical_and(~pred, true).sum()
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else float((2 * tp) / denom)
        per_class_f1[name] = f1
        f1_values.append(f1)

    return {
        "exact_match_accuracy": float(exact.mean()),
        "hamming_accuracy": float(hamming.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class_f1": per_class_f1,
    }

def write_trace(trace_rows, path):
    if not trace_rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)

def parse_args():
    config_parser, config = load_config_from_args("Internvl3")
    parser = argparse.ArgumentParser(
        description="InternVL3-8B label-level DDM-VLM wafer-map prediction.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--test-path",
        default=config_get(config, "data.test_path"),
        help="Path to the test .npz file.",
    )
    parser.add_argument(
        "--database-path",
        default=config_get(config, "data.database_path"),
        help="Path to the retrieval database .npz file.",
    )
    parser.add_argument(
        "--index-path",
        default=config_get(config, "retrieval.index_path"),
        help="Path to the FAISS index file.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=int(config_get(config, "ddm_vlm.max_rounds", 3)),
    )
    parser.add_argument(
        "--retrieval-split",
        type=parse_split,
        default=config_get(config, "ddm_vlm.retrieval_split", [4, 3, 3]),
    )
    parser.add_argument(
        "--example-mode",
        choices=["split", "cumulative"],
        default=config_get(config, "ddm_vlm.example_mode", "split"),
        help="Use only new examples each round, or all examples seen so far.",
    )
    parser.add_argument(
        "--boundary",
        type=float,
        default=float(config_get(config, "ddm_vlm.boundary", 2.0)),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=float(config_get(config, "ddm_vlm.alpha", 0.8)),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=float(config_get(config, "ddm_vlm.beta", 0.2)),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=float(config_get(config, "ddm_vlm.sigma", 0.05)),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=float(config_get(config, "ddm_vlm.tau", 0.05)),
    )
    parser.add_argument(
        "--evidence-clip",
        type=float,
        default=float(config_get(config, "ddm_vlm.evidence_clip", 3.0)),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config_get(config, "ddm_vlm.seed", 42)),
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=bool(config_get(config, "ddm_vlm.early_stop", True)),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(config_get(config, "ddm_vlm.max_new_tokens", 8)),
    )
    parser.add_argument(
        "--output-dir",
        default=config_get(config, "outputs.ddm_vlm_dir"),
        help="Output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick smoke tests.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    resources = load_resources(args)
    test_data = np.load(args.test_path)
    test_maps = test_data["arr_0"]
    test_labels = test_data["arr_1"].astype(np.int32)

    if args.limit is not None:
        test_maps = test_maps[: args.limit]
        test_labels = test_labels[: args.limit]

    InternVL3.initialize(multi_turn=False)

    rng = np.random.default_rng(args.seed)
    predictions = []
    trace_rows = []
    rounds_used = []

    for idx in tqdm(range(len(test_maps))):
        pred, sample_trace, sample_rounds = predict_one(
            test_maps[idx],
            resources,
            args,
            rng,
            sample_index=idx,
        )
        predictions.append(pred)
        trace_rows.extend(sample_trace)
        rounds_used.append(sample_rounds)

        correct = np.array_equal(pred, test_labels[idx])
        print(f"Sample {idx}: {correct}")

    predictions = np.asarray(predictions, dtype=np.int32)

    result_csv = os.path.join(args.output_dir, "final_result.csv")
    trace_csv = os.path.join(args.output_dir, "trace.csv")
    summary_json = os.path.join(args.output_dir, "summary.json")

    pd.DataFrame(predictions).to_csv(result_csv, index=False)
    write_trace(trace_rows, trace_csv)

    summary = {
        "model": "InternVL3",
        "test_path": args.test_path,
        "database_path": args.database_path,
        "index_path": args.index_path,
        "num_samples": int(len(test_maps)),
        "defects": DEFECTS,
        "valid_label_count": int(len(resources["valid_labels"])),
        "hyperparameters": {
            "max_rounds": args.max_rounds,
            "retrieval_split": args.retrieval_split,
            "example_mode": args.example_mode,
            "boundary": args.boundary,
            "alpha": args.alpha,
            "beta": args.beta,
            "sigma": args.sigma,
            "tau": args.tau,
            "evidence_clip": args.evidence_clip,
            "seed": args.seed,
            "early_stop": args.early_stop,
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": compute_metrics(predictions, test_labels),
        "avg_rounds_used": float(np.mean(rounds_used)),
        "avg_vlm_calls": float(np.mean(rounds_used)),
        "result_csv": result_csv,
        "trace_csv": trace_csv,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
