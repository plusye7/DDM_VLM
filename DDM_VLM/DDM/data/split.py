from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_PATH = ""
DEFAULT_OUTPUT_DIR = ""
DEFAULT_RATIOS = ("9:1", "7:3")

DEFECT_TO_INDEX = {
    "C": 0,
    "D": 1,
    "EL": 2,
    "ER": 3,
    "L": 4,
    "NF": 5,
    "S": 6,
    "R": 7,
}

CATEGORY_SPECS = (
    ("C1_Normal", ()),
    ("C2_Center", ("C",)),
    ("C3_Donut", ("D",)),
    ("C4_Edge_Loc", ("EL",)),
    ("C5_Edge_Ring", ("ER",)),
    ("C6_Loc", ("L",)),
    ("C7_Near_Full", ("NF",)),
    ("C8_Scratch", ("S",)),
    ("C9_Random", ("R",)),
    ("C10_C+EL", ("C", "EL")),
    ("C11_C+ER", ("C", "ER")),
    ("C12_C+L", ("C", "L")),
    ("C13_C+S", ("C", "S")),
    ("C14_D+EL", ("D", "EL")),
    ("C15_D+ER", ("D", "ER")),
    ("C16_D+L", ("D", "L")),
    ("C17_D+S", ("D", "S")),
    ("C18_EL+L", ("EL", "L")),
    ("C19_EL+S", ("EL", "S")),
    ("C20_ER+L", ("ER", "L")),
    ("C21_ER+S", ("ER", "S")),
    ("C22_L+S", ("L", "S")),
    ("C23_C+EL+L", ("C", "EL", "L")),
    ("C24_C+EL+S", ("C", "EL", "S")),
    ("C25_C+ER+L", ("C", "ER", "L")),
    ("C26_C+ER+S", ("C", "ER", "S")),
    ("C27_C+L+S", ("C", "L", "S")),
    ("C28_D+EL+L", ("D", "EL", "L")),
    ("C29_D+EL+S", ("D", "EL", "S")),
    ("C30_D+ER+L", ("D", "ER", "L")),
    ("C31_D+ER+S", ("D", "ER", "S")),
    ("C32_D+L+S", ("D", "L", "S")),
    ("C33_EL+L+S", ("EL", "L", "S")),
    ("C34_ER+L+S", ("ER", "L", "S")),
    ("C35_C+L+EL+S", ("C", "L", "EL", "S")),
    ("C36_C+L+ER+S", ("C", "L", "ER", "S")),
    ("C37_D+L+EL+S", ("D", "L", "EL", "S")),
    ("C38_D+L+ER+S", ("D", "L", "ER", "S")),
)


@dataclass(frozen=True)
class SplitRatio:
    raw: str
    database_part: int
    test_part: int

    @property
    def suffix(self) -> str:
        return f"{self.database_part}_{self.test_part}"

    @property
    def total_part(self) -> int:
        return self.database_part + self.test_part


def parse_ratio(raw: str) -> SplitRatio:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio {raw!r}. Use the form database:test, for example 9:1."
        )

    left, right = raw.split(":", maxsplit=1)
    try:
        database_part = int(left)
        test_part = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio {raw!r}. Ratio parts must be integers."
        ) from exc

    if database_part <= 0 or test_part <= 0:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio {raw!r}. Ratio parts must be positive."
        )

    return SplitRatio(raw=raw, database_part=database_part, test_part=test_part)


def category_label(defects: tuple[str, ...]) -> np.ndarray:
    import numpy as np

    label = np.zeros(len(DEFECT_TO_INDEX), dtype=np.int32)
    for defect in defects:
        label[DEFECT_TO_INDEX[defect]] = 1
    return label


def collect_category_indices(labels: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    import numpy as np

    if labels.ndim != 2 or labels.shape[1] != len(DEFECT_TO_INDEX):
        raise ValueError(
            f"Expected arr_1 to have shape (N, {len(DEFECT_TO_INDEX)}), "
            f"got {labels.shape}."
        )

    used = np.zeros(labels.shape[0], dtype=bool)
    category_indices = []

    for category_name, defects in CATEGORY_SPECS:
        label = category_label(defects)
        indices = np.flatnonzero(np.all(labels == label, axis=1))
        if indices.size == 0:
            raise ValueError(f"No samples found for category {category_name}.")
        category_indices.append((category_name, label, indices))
        used[indices] = True

    if not np.all(used):
        missing = int(np.count_nonzero(~used))
        raise ValueError(f"{missing} samples do not match the 38 known categories.")

    return category_indices


def split_one_ratio(
    maps: np.ndarray,
    labels: np.ndarray,
    category_indices: list[tuple[str, np.ndarray, np.ndarray]],
    ratio: SplitRatio,
    rng: np.random.Generator,
    seed: int,
    output_dir: Path,
    shuffle: bool,
    compressed: bool,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, object]:
    import numpy as np

    database_chunks = []
    database_label_chunks = []
    test_chunks = []
    test_label_chunks = []
    database_ranges = {}
    test_ranges = {}
    categories = []
    database_start = 0
    test_start = 0

    for category_name, label, indices in category_indices:
        split_indices = indices.copy()
        if shuffle:
            rng.shuffle(split_indices)

        test_count = split_indices.size * ratio.test_part // ratio.total_part
        database_count = split_indices.size - test_count
        database_indices = split_indices[:database_count]
        test_indices = split_indices[database_count:]

        database_chunks.append(maps[database_indices])
        database_label_chunks.append(labels[database_indices])
        test_chunks.append(maps[test_indices])
        test_label_chunks.append(labels[test_indices])

        database_end = database_start + database_count - 1
        test_end = test_start + test_count - 1
        database_ranges[category_name] = [int(database_start), int(database_end)]
        test_ranges[category_name] = [int(test_start), int(test_end)]
        categories.append(
            {
                "name": category_name,
                "label": label.astype(int).tolist(),
                "total": int(split_indices.size),
                "database": int(database_count),
                "test": int(test_count),
            }
        )
        database_start += database_count
        test_start += test_count

    database_maps = np.concatenate(database_chunks, axis=0)
    database_labels = np.concatenate(database_label_chunks, axis=0)
    test_maps = np.concatenate(test_chunks, axis=0)
    test_labels = np.concatenate(test_label_chunks, axis=0)

    database_path = output_dir / f"Wafer_Map_Datasets_database_{ratio.suffix}.npz"
    test_path = output_dir / f"Wafer_Map_Datasets_test_{ratio.suffix}.npz"
    database_ranges_path = output_dir / f"database_index_ranges_{ratio.suffix}.json"
    test_ranges_path = output_dir / f"test_index_ranges_{ratio.suffix}.json"
    summary_path = output_dir / f"split_summary_{ratio.suffix}.json"

    outputs = (
        database_path,
        test_path,
        database_ranges_path,
        test_ranges_path,
        summary_path,
    )
    if not dry_run and not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --overwrite to replace them:\n"
                + "\n".join(existing)
            )

    summary = {
        "ratio": ratio.raw,
        "database_path": str(database_path),
        "test_path": str(test_path),
        "database_count": int(database_maps.shape[0]),
        "test_count": int(test_maps.shape[0]),
        "total_count": int(maps.shape[0]),
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "categories": categories,
        "count_rule": "test_count = total_count * test_part // (database_part + test_part); database_count = total_count - test_count",
    }

    if dry_run:
        return summary

    savez = np.savez_compressed if compressed else np.savez
    savez(database_path, database_maps, database_labels)
    savez(test_path, test_maps, test_labels)

    database_ranges_path.write_text(
        json.dumps(database_ranges, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    test_ranges_path.write_text(
        json.dumps(test_ranges, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified MixedWM38 query-database/test splits in the same "
            "38-category order used by the existing 8:2 files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the source MixedWM38-style .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated split files.",
    )
    parser.add_argument(
        "--ratios",
        type=parse_ratio,
        nargs="+",
        default=[parse_ratio(raw) for raw in DEFAULT_RATIOS],
        help="One or more database:test ratios, for example --ratios 9:1 7:3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling within each category.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep the original order within each category instead of shuffling.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed instead of np.savez.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print split counts; do not write files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated files.",
    )
    args = parser.parse_args()
    if not args.input:
        parser.error("--input is required.")
    if not args.output_dir:
        parser.error("--output-dir is required.")
    return args


def main() -> None:
    args = parse_args()
    import numpy as np

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(input_path, allow_pickle=True)
    maps = data["arr_0"]
    labels = data["arr_1"].astype(np.int32)

    if maps.shape[0] != labels.shape[0]:
        raise ValueError(
            f"arr_0 and arr_1 contain different numbers of samples: "
            f"{maps.shape[0]} vs {labels.shape[0]}."
        )

    category_indices = collect_category_indices(labels)

    print(f"Input: {input_path}")
    print(f"Total samples: {maps.shape[0]}")
    print(f"Shuffle within each category: {not args.no_shuffle}")
    print(f"Seed: {args.seed}")

    for ratio in args.ratios:
        rng = np.random.default_rng(args.seed)
        summary = split_one_ratio(
            maps=maps,
            labels=labels,
            category_indices=category_indices,
            ratio=ratio,
            rng=rng,
            seed=args.seed,
            output_dir=output_dir,
            shuffle=not args.no_shuffle,
            compressed=args.compressed,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

        print(
            f"{ratio.raw} -> database: {summary['database_count']}, "
            f"test: {summary['test_count']}"
        )
        if not args.dry_run:
            print(f"  saved database: {summary['database_path']}")
            print(f"  saved test: {summary['test_path']}")


if __name__ == "__main__":
    main()
