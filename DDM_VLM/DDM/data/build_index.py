from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATABASE_PATH = ""
DEFAULT_OUTPUT_DIR = ""
DEFAULT_INDEX_NAME = "global_faiss_cosine.bin"
DEFAULT_LABELS_NAME = "global_labels.npy"
DEFAULT_LABEL_VECTORS_NAME = "global_label_vectors.npy"
DEFAULT_METADATA_NAME = "global_faiss_metadata.json"

DEFECT_NAMES = (
    "center",
    "donut",
    "edge_loc",
    "edge_ring",
    "loc",
    "near_full",
    "scratch",
    "random",
)

TORCH_MODULE = None
TORCH_DEVICE = None


def ensure_torch():
    global TORCH_MODULE
    if TORCH_MODULE is None:
        import torch

        TORCH_MODULE = torch
    return TORCH_MODULE


def get_torch_device() -> str:
    global TORCH_DEVICE
    if TORCH_DEVICE is None:
        torch = ensure_torch()
        TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return TORCH_DEVICE


def load_clip_model():
    device = get_torch_device()
    import clip

    model, preprocess = clip.load("ViT-B/32", device=device)
    return model.eval(), preprocess


def process_wafer_map(wafer: np.ndarray, min_cluster_size: int = 10) -> Image.Image:
    """Convert a wafer map into a filtered RGB defect image."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

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


def extract_feature(
    image: Image.Image,
    model,
    preprocess,
) -> np.ndarray:
    """Extract one L2-normalized CLIP feature vector from a wafer image."""
    import numpy as np

    torch = ensure_torch()
    device = get_torch_device()
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(tensor)

    vector = feature.view(feature.size(0), -1).cpu().numpy()[0]
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Feature extractor returned a zero vector.")
    return (vector / norm).astype(np.float32)


def label_to_text(label: np.ndarray) -> str:
    names = [name for name, value in zip(DEFECT_NAMES, label) if int(value) == 1]
    if not names:
        return "normal"
    return "+".join(names)


def load_database(database_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    data = np.load(database_path)
    maps = data["arr_0"]
    labels = data["arr_1"].astype(np.int32)

    if maps.shape[0] != labels.shape[0]:
        raise ValueError(
            "arr_0 and arr_1 contain different numbers of samples: "
            f"{maps.shape[0]} vs {labels.shape[0]}."
        )
    if labels.ndim != 2 or labels.shape[1] != len(DEFECT_NAMES):
        raise ValueError(
            f"Expected arr_1 to have shape (N, {len(DEFECT_NAMES)}), "
            f"got {labels.shape}."
        )

    return maps, labels


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    return {
        "index": output_dir / args.index_name,
        "labels": output_dir / args.labels_name,
        "label_vectors": output_dir / args.label_vectors_name,
        "metadata": output_dir / args.metadata_name,
    }


def ensure_outputs_can_be_written(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            f"{formatted}"
        )


def build_index(args: argparse.Namespace) -> dict[str, object]:
    import faiss
    import numpy as np

    database_path = Path(args.database_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = output_paths(args)
    paths = {name: path.resolve() for name, path in paths.items()}
    ensure_outputs_can_be_written(paths, args.overwrite)

    maps, labels = load_database(database_path)
    model, preprocess = load_clip_model()

    vectors = []
    for idx, wafer in enumerate(maps):
        image = process_wafer_map(wafer, min_cluster_size=args.min_cluster_size)
        vector = extract_feature(image, model, preprocess)
        vectors.append(vector)

        if args.report_every > 0 and (idx + 1) % args.report_every == 0:
            print(f"Embedded {idx + 1}/{len(maps)} samples.")

    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    label_texts = np.asarray([label_to_text(label) for label in labels])
    faiss.write_index(index, str(paths["index"]))
    np.save(paths["labels"], label_texts)
    np.save(paths["label_vectors"], labels)

    metadata = {
        "database_path": str(database_path),
        "index_path": str(paths["index"]),
        "labels_path": str(paths["labels"]),
        "label_vectors_path": str(paths["label_vectors"]),
        "feature_model": "clip",
        "clip_model": "ViT-B/32",
        "feature_dim": int(vectors.shape[1]),
        "sample_count": int(vectors.shape[0]),
        "min_cluster_size": int(args.min_cluster_size),
        "distance": "cosine",
        "faiss_index": "IndexFlatIP",
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS cosine index from a wafer retrieval database."
    )
    parser.add_argument(
        "--database-path",
        default=DEFAULT_DATABASE_PATH,
        help="Path to the retrieval database .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated FAISS index and label files.",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Output filename for the FAISS index.",
    )
    parser.add_argument(
        "--labels-name",
        default=DEFAULT_LABELS_NAME,
        help="Output filename for text labels aligned with index rows.",
    )
    parser.add_argument(
        "--label-vectors-name",
        default=DEFAULT_LABEL_VECTORS_NAME,
        help="Output filename for one-hot labels aligned with index rows.",
    )
    parser.add_argument(
        "--metadata-name",
        default=DEFAULT_METADATA_NAME,
        help="Output filename for build metadata.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum connected defect component size kept during preprocessing.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=500,
        help="Print progress every N samples. Use 0 to disable progress messages.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated files.",
    )
    args = parser.parse_args()

    if not args.database_path:
        parser.error("--database-path is required.")
    if not args.output_dir:
        parser.error("--output-dir is required.")

    return args


def main() -> None:
    metadata = build_index(parse_args())
    print("FAISS index built successfully.")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
