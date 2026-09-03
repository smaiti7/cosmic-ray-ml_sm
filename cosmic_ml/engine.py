from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def multitask_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    energy_weight: float = 1.0,
    core_weight: float = 1.0,
    direction_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    classification = F.binary_cross_entropy_with_logits(predictions["class_logit"], batch["y_class"])
    signal = batch["y_class"] > 0.5
    zero = classification.new_zeros(())
    energy = F.smooth_l1_loss(predictions["energy"][signal], batch["y_energy"][signal]) if signal.any() else zero
    core = F.smooth_l1_loss(predictions["core"][signal], batch["y_core"][signal]) if signal.any() else zero
    direction = zero
    if "direction" in predictions and signal.any():
        direction = (1.0 - F.cosine_similarity(predictions["direction"][signal], batch["y_direction"][signal])).mean()
    total = classification + energy_weight * energy + core_weight * core + direction_weight * direction
    parts = {
        "loss": float(total.detach()),
        "classification": float(classification.detach()),
        "energy": float(energy.detach()),
        "core": float(core.detach()),
        "direction": float(direction.detach()),
    }
    return total, parts


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    energy_weight: float = 1.0,
    core_weight: float = 1.0,
    direction_weight: float = 0.2,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    context = nullcontext if training else torch.inference_mode
    with context():
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions = model(batch["waveform"], batch["mask"])
                loss, parts = multitask_loss(
                    predictions,
                    batch,
                    energy_weight=energy_weight,
                    core_weight=core_weight,
                    direction_weight=direction_weight,
                )
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            batch_size = len(batch["y_class"])
            examples += batch_size
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
    return {name: value / examples for name, value in totals.items()}


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["waveform"], batch["mask"])
        values = {
            "index": batch["index"],
            "y_class": batch["y_class"],
            "y_energy": batch["y_energy"],
            "y_core": batch["y_core"],
            "y_direction": batch["y_direction"],
            "class_probability": output["class_logit"].sigmoid(),
            "energy": output["energy"],
            "core": output["core"],
        }
        if "direction" in output:
            values["direction"] = output["direction"]
        for name, value in values.items():
            collected.setdefault(name, []).append(value.detach().cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}

