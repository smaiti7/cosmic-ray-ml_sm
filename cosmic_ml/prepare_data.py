from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from .data import signed_log1p


EXPECTED_FILES = {
    "X.npy",
    "positions.npy",
    "y_class.npy",
    "y_energy.npy",
    "y_core.npy",
    "y_direction.npy",
    "detector_masks.npy",
}


def extract_archive(archive_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    if all((extracted_dir / name).exists() for name in EXPECTED_FILES):
        print(f"Using existing extracted arrays in {extracted_dir}")
        return
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing {archive_path}. Run scripts/download_data.sh first."
        )
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if names != EXPECTED_FILES:
            raise ValueError(f"Unexpected archive members: {sorted(names)}")
        print(f"Extracting {archive_path} to {extracted_dir} ...")
        archive.extractall(extracted_dir)


def stratified_splits(
    labels: np.ndarray, seed: int, train_fraction: float, val_fraction: float
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_train = round(len(indices) * train_fraction)
        n_val = round(len(indices) * val_fraction)
        groups["train"].append(indices[:n_train])
        groups["val"].append(indices[n_train : n_train + n_val])
        groups["test"].append(indices[n_train + n_val :])
    result: dict[str, np.ndarray] = {}
    for name, parts in groups.items():
        result[name] = np.concatenate(parts).astype(np.int64)
        rng.shuffle(result[name])
    return result


def compute_input_stats(
    X: np.ndarray, masks: np.ndarray, train_indices: np.ndarray, chunk_size: int
) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    ordered = np.sort(train_indices)
    for start in range(0, len(ordered), chunk_size):
        ids = ordered[start : start + chunk_size]
        values = signed_log1p(np.asarray(X[ids], dtype=np.float32))
        active = np.asarray(masks[ids], dtype=np.float32)[:, None, :]
        total += float(np.sum(values * active, dtype=np.float64))
        total_sq += float(np.sum(values * values * active, dtype=np.float64))
        count += int(active.sum()) * X.shape[1]
        if start == 0 or start + chunk_size >= len(ordered) or start % (20 * chunk_size) == 0:
            print(f"  normalization pass: {min(start + chunk_size, len(ordered))}/{len(ordered)}")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return mean, variance**0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract data, split it, and compute train-only statistics.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.val_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    archive_path = args.data_dir / "cosmic_array_dataset.npz"
    extracted = args.data_dir / "extracted"
    processed = args.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    extract_archive(archive_path, extracted)

    stats_path = processed / "stats.json"
    splits_path = processed / "splits.npz"
    if stats_path.exists() and splits_path.exists() and not args.force:
        print(f"Prepared data already exists in {processed}; pass --force to recompute.")
        return

    X = np.load(extracted / "X.npy", mmap_mode="r")
    masks = np.load(extracted / "detector_masks.npy", mmap_mode="r")
    labels = np.load(extracted / "y_class.npy")
    energy = np.load(extracted / "y_energy.npy")
    core = np.load(extracted / "y_core.npy")
    positions = np.load(extracted / "positions.npy")

    if X.shape != (len(labels), X.shape[1], len(positions)):
        raise ValueError(f"Inconsistent shapes: X={X.shape}, labels={labels.shape}, positions={positions.shape}")
    if not np.array_equal(np.unique(masks), np.array([0.0, 1.0], dtype=masks.dtype)):
        raise ValueError("Detector mask is not binary.")

    splits = stratified_splits(labels, args.seed, args.train_fraction, args.val_fraction)
    np.savez(splits_path, **splits)
    train = splits["train"]
    train_signal = train[labels[train] == 1]
    input_mean, input_std = compute_input_stats(X, masks, train, args.chunk_size)

    stats = {
        "seed": args.seed,
        "shape": {"events": int(X.shape[0]), "time_bins": int(X.shape[1]), "detectors": int(X.shape[2])},
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "input": {"transform": "signed_log1p", "mean": input_mean, "std": input_std},
        "positions": {"mean": positions.mean(axis=0).tolist(), "std": positions.std(axis=0).tolist()},
        "energy": {"mean": float(energy[train_signal].mean()), "std": float(energy[train_signal].std())},
        "core": {"mean": core[train_signal].mean(axis=0).tolist(), "std": core[train_signal].std(axis=0).tolist()},
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "active_detectors": {
            "min": int(masks.sum(axis=1).min()),
            "mean": float(masks.sum(axis=1).mean()),
            "max": int(masks.sum(axis=1).max()),
        },
    }
    with stats_path.open("w") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(json.dumps(stats, indent=2))
    print(f"Saved {splits_path} and {stats_path}")


if __name__ == "__main__":
    main()

