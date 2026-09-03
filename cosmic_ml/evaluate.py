from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from torch.utils.data import DataLoader

from .data import CosmicDataset, load_splits, load_stats, normalized_positions
from .engine import predict
from .metrics import compute_metrics, regression_to_physical
from .model import build_model
from .train import choose_device


def plot_results(predictions: dict[str, np.ndarray], stats: dict, output_dir: Path) -> None:
    values = regression_to_physical(predictions, stats)
    labels = values["y_class"].astype(int)
    probability = values["class_probability"]
    signal = labels == 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(labels, probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(labels, probability, ax=axes[1])
    fig.tight_layout()
    fig.savefig(output_dir / "classification_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hexbin(values["y_energy"][signal], values["energy"][signal], gridsize=45, mincnt=1)
    limits = [float(values["y_energy"][signal].min()), float(values["y_energy"][signal].max())]
    axes[0].plot(limits, limits, "r--", linewidth=1)
    axes[0].set(xlabel="True log-energy", ylabel="Predicted log-energy", title="Energy reconstruction")
    distance = np.linalg.norm(values["core"][signal] - values["y_core"][signal], axis=1)
    axes[1].hist(distance, bins=50, histtype="step", linewidth=1.5)
    axes[1].set(xlabel="Core distance error", ylabel="Events", title="Core reconstruction")
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model and save metrics/plots.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int, help="Optional deterministic subset for a quick demo.")
    args = parser.parse_args()

    device = choose_device(args.device)
    stats = load_stats(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = dict(checkpoint["model_config"])
    model_type = str(config.pop("model_type", "original"))
    positions = torch.from_numpy(normalized_positions(args.data_dir, stats))
    model = build_model(model_type, positions, **config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    indices = load_splits(args.data_dir)[args.split][: args.max_samples]
    dataset = CosmicDataset(indices, args.data_dir, stats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(f"Evaluating {len(dataset):,} {args.split} events on {device} ...")
    predictions = predict(model, loader, device)
    metrics = compute_metrics(predictions, stats)

    output_dir = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    plot_results(predictions, stats, output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()

