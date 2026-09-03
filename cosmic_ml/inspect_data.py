from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Print dataset checks and create initial EDA figures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = args.data_dir / "extracted"

    X = np.load(extracted / "X.npy", mmap_mode="r")
    positions = np.load(extracted / "positions.npy")
    labels = np.load(extracted / "y_class.npy")
    energy = np.load(extracted / "y_energy.npy")
    core = np.load(extracted / "y_core.npy")
    direction = np.load(extracted / "y_direction.npy")
    masks = np.load(extracted / "detector_masks.npy", mmap_mode="r")
    classes, counts = np.unique(labels, return_counts=True)
    summary = {
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "class_counts": {str(int(k)): int(v) for k, v in zip(classes, counts)},
        "position_min": positions.min(axis=0).tolist(),
        "position_max": positions.max(axis=0).tolist(),
        "active_detectors_min_mean_max": [
            float(masks.sum(axis=1).min()),
            float(masks.sum(axis=1).mean()),
            float(masks.sum(axis=1).max()),
        ],
        "signal_energy_min_mean_max": [
            float(energy[labels == 1].min()),
            float(energy[labels == 1].mean()),
            float(energy[labels == 1].max()),
        ],
        "signal_direction_norm_mean": float(np.linalg.norm(direction[labels == 1], axis=1).mean()),
        "background_targets_are_nan": bool(np.isnan(energy[labels == 0]).all() and np.isnan(core[labels == 0]).all()),
    }
    print(json.dumps(summary, indent=2))
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(positions[:, 0], positions[:, 1], s=18)
    ax.set(xlabel="x", ylabel="y", title=f"Detector geometry ({len(positions)} sensors)", aspect="equal")
    fig.tight_layout()
    fig.savefig(args.output_dir / "detector_geometry.png", dpi=160)
    plt.close(fig)

    event_ids = [int(np.flatnonzero(labels == value)[0]) for value in (0, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (event_id, name) in enumerate(zip(event_ids, ("Background", "Signal"))):
        event = np.asarray(X[event_id])
        charge = np.maximum(event, 0).sum(axis=0)
        scatter = axes[row, 0].scatter(positions[:, 0], positions[:, 1], c=np.log1p(charge), s=35)
        axes[row, 0].set(title=f"{name}: log(1 + detector charge)", aspect="equal")
        fig.colorbar(scatter, ax=axes[row, 0])
        image = axes[row, 1].imshow(event, aspect="auto", origin="lower", cmap="viridis")
        axes[row, 1].set(xlabel="Detector index", ylabel="Time bin", title=f"{name}: raw waveform matrix")
        fig.colorbar(image, ax=axes[row, 1])
    fig.tight_layout()
    fig.savefig(args.output_dir / "example_events.png", dpi=160)
    plt.close(fig)
    print(f"Saved EDA outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

