import argparse
import csv
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from DDM.config import config_get, load_config_from_args
from DDM.models.Owl3.Owl3 import OWL3

EXPERIMENT_NAME = "direct_vlm"

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

def process_wafer_map(wafer, min_cluster_size=10):
    """
    Use exactly the same wafer rendering as the RAG/DDM-VLM experiments:
    remove connected defect components smaller than min_cluster_size,
    render retained defects in red and all other cells in black,
    then resize to 384x384 with nearest-neighbor interpolation.
    """
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

def create_prompt():
    description_lines = [
        f"- {name}: {DEFECT_DESCRIPTIONS[name]}"
        for name in DEFECTS
    ]
    description_text = "\n".join(description_lines)

    return f"""
Role: You are a professional semiconductor yield engineer.

I will give you one wafer image. Black pixels represent normal areas and red
pixels represent defective areas. A wafer may contain zero, one, or multiple
defect patterns.

Your task is to determine which of the eight defect patterns are present in
the target wafer.

Defect descriptions:
{description_text}

Important constraints:
- Analyze the target wafer using defect location, shape, distribution,
  continuity, and coverage.
- Judge the complete defect composition of the wafer.
- Return an eight-dimensional multi-hot vector in exactly this order:
  [center, donut, edge_loc, edge_ring, loc, near_full, scratch, random]
- Use 1 when a defect pattern is present and 0 when it is absent.
- If no defect pattern is present, return [0, 0, 0, 0, 0, 0, 0, 0].
- Output only the eight-dimensional multi-hot vector. Do not output any
  explanation or additional text.

Final answer format:
[0, 0, 0, 0, 0, 0, 0, 0]
"""

def parse_multihot(raw_output):
    """
    Parse an 8-dimensional binary vector.

    Primary path:
        find a bracketed vector such as [1, 0, 0, 0, 1, 0, 1, 0]

    Fallback:
        if the entire generated text contains exactly eight standalone
        binary values, accept them.
    """
    text = str(raw_output).strip()

    bracket_matches = re.findall(r"\[[^\[\]]*\]", text)
    for candidate in bracket_matches:
        values = re.findall(r"\b[01]\b", candidate)
        if len(values) == len(DEFECTS):
            return [int(v) for v in values], "bracketed_multihot"

    values = re.findall(r"\b[01]\b", text)
    if len(values) == len(DEFECTS):
        return [int(v) for v in values], "binary_fallback"

    return None, "missing"

def predict(image, max_new_tokens):
    image_input = process_wafer_map(image)
    prompt = create_prompt()

    # Keep every test wafer as an independent one-shot interaction.
    OWL3.reset_chat()

    result = OWL3.predict(
        prompt,
        image_input,
        example_texts=None,
        example_images=None,
        max_new_tokens=max_new_tokens,
    )

    raw_output = result.get("raw_output", "")
    label, parse_source = parse_multihot(raw_output)

    parse_failed = label is None
    if parse_failed:
        # Invalid model output is counted as an incorrect prediction.
        label = [-1] * len(DEFECTS)

    return np.array(label, dtype=np.int32), {
        "raw_output": raw_output,
        "parse_source": parse_source,
        "parse_failed": parse_failed,
        "num_generated_tokens": result.get("num_generated_tokens"),
    }

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
    config_parser, config = load_config_from_args("Owl3")
    parser = argparse.ArgumentParser(
        description=(
            "Owl3 direct-VLM wafer map multi-label prediction without "
            "retrieval, candidate-label restriction, or DDM accumulation."
        ),
        parents=[config_parser],
    )

    parser.add_argument(
        "--test-path",
        default=config_get(config, "data.test_path"),
        help="Path to the test .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        default=config_get(config, "outputs.direct_vlm_dir"),
        help="Output directory.",
    )

    # An 8-dimensional vector may require substantially more than 8 tokens.
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(config_get(config, "direct_vlm.max_new_tokens", 32)),
    )
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    test_data = np.load(args.test_path)
    test_maps = test_data["arr_0"]
    test_labels = test_data["arr_1"].astype(np.int32)

    if args.limit is not None:
        test_maps = test_maps[: args.limit]
        test_labels = test_labels[: args.limit]

    OWL3.initialize(multi_turn=False)

    predictions = []
    trace_rows = []

    for idx in tqdm(range(len(test_maps))):
        pred, trace = predict(
            test_maps[idx],
            args.max_new_tokens,
        )
        predictions.append(pred)

        correct = np.array_equal(pred, test_labels[idx])

        trace_rows.append(
            {
                "index": idx,
                "raw_output": trace["raw_output"],
                "parse_source": trace["parse_source"],
                "parse_failed": trace["parse_failed"],
                "prediction": json.dumps(as_jsonable(pred)),
                "label": json.dumps(as_jsonable(test_labels[idx])),
                "correct": correct,
                "num_generated_tokens": trace["num_generated_tokens"],
            }
        )

        print(f"Sample {idx}: {correct}")

    predictions = np.asarray(predictions, dtype=np.int32)

    result_csv = os.path.join(args.output_dir, "final_result.csv")
    trace_csv = os.path.join(args.output_dir, "trace.csv")
    summary_json = os.path.join(args.output_dir, "summary.json")

    pd.DataFrame(
        predictions,
        columns=DEFECTS,
    ).to_csv(
        result_csv,
        index=False,
    )

    write_trace(trace_rows, trace_csv)

    parse_failed_count = int(
        sum(row["parse_failed"] for row in trace_rows)
    )

    summary = {
        "model": "Owl3",
        "scheme": EXPERIMENT_NAME,
        "task_formulation": "free_8bit_multihot",
        "test_path": args.test_path,
        "num_samples": int(len(test_maps)),
        "defects": DEFECTS,
        "uses_retrieval": False,
        "uses_candidate_label_restriction": False,
        "uses_ddm": False,
        "avg_vlm_calls": 1.0,
        "hyperparameters": {
            "min_cluster_size": 10,
            "image_size": [384, 384],
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": compute_metrics(predictions, test_labels),
        "parse_failed_count": parse_failed_count,
        "parse_failure_rate": (
            float(parse_failed_count / len(test_maps))
            if len(test_maps) > 0
            else 0.0
        ),
        "result_csv": result_csv,
        "trace_csv": trace_csv,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
