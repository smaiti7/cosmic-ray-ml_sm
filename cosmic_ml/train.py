from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CosmicDataset, load_splits, load_stats, normalized_positions
from .engine import run_epoch
from .model import build_model, model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Original or Attention temporal Deep Sets model.")
    parser.add_argument("--model-type", choices=("original", "attention"), default="original")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deepset"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Use FP16 mixed precision (benchmark on GTX 1080 Ti).")
    parser.add_argument("--predict-direction", action="store_true")
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--core-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=0.2)
    parser.add_argument("--temporal-dim", type=int, default=48)
    parser.add_argument("--detector-dim", type=int, default=64)
    parser.add_argument("--event-dim", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check nvidia-smi and the PyTorch CUDA build.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    indices: np.ndarray,
    data_dir: Path,
    stats: dict,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    dataset = CosmicDataset(indices, data_dir=data_dir, stats=stats)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.data_dir)
    splits = load_splits(args.data_dir)
    train_indices = splits["train"][: args.max_train_samples]
    val_indices = splits["val"][: args.max_val_samples]
    train_loader = make_loader(
        train_indices, args.data_dir, stats, args.batch_size, args.num_workers, True, device, args.seed
    )
    val_loader = make_loader(
        val_indices, args.data_dir, stats, args.batch_size, args.num_workers, False, device, args.seed
    )

    positions = torch.from_numpy(normalized_positions(args.data_dir, stats))
    model = build_model(
        args.model_type,
        positions,
        temporal_dim=args.temporal_dim,
        detector_dim=args.detector_dim,
        event_dim=args.event_dim,
        predict_direction=args.predict_direction,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; AMP: {use_amp}; parameters: {parameter_count:,}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}; VRAM: {props.total_memory / 2**30:.1f} GiB")
    print(f"Events: train={len(train_indices):,}, validation={len(val_indices):,}")

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            energy_weight=args.energy_weight,
            core_weight=args.core_weight,
            direction_weight=args.direction_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            energy_weight=args.energy_weight,
            core_weight=args.core_weight,
            direction_weight=args.direction_weight,
        )
        scheduler.step(val_metrics["loss"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
        }
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train {train_metrics['loss']:.4f} | "
            f"val {val_metrics['loss']:.4f} | lr {row['learning_rate']:.2e} | {row['seconds']:.1f}s"
        )

        if val_metrics["loss"] < best_loss - 1e-5:
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model_config(model),
                "stats": stats,
                "epoch": epoch,
                "val_loss": best_loss,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")
            print(f"  saved {args.output_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1

        with (args.output_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print(f"Training completed in {(time.time() - started) / 60:.1f} min; best val loss={best_loss:.4f}")


if __name__ == "__main__":
    main()

