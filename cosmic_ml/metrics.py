from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def regression_to_physical(predictions: dict[str, np.ndarray], stats: dict[str, Any]) -> dict[str, np.ndarray]:
    result = dict(predictions)
    energy_mean = float(stats["energy"]["mean"])
    energy_std = float(stats["energy"]["std"])
    core_mean = np.asarray(stats["core"]["mean"])
    core_std = np.asarray(stats["core"]["std"])
    result["energy"] = result["energy"] * energy_std + energy_mean
    result["y_energy"] = result["y_energy"] * energy_std + energy_mean
    result["core"] = result["core"] * core_std + core_mean
    result["y_core"] = result["y_core"] * core_std + core_mean
    return result


def compute_metrics(predictions: dict[str, np.ndarray], stats: dict[str, Any]) -> dict[str, float]:
    values = regression_to_physical(predictions, stats)
    labels = values["y_class"].astype(int)
    probability = values["class_probability"]
    metrics = {
        "classification/roc_auc": float(roc_auc_score(labels, probability)),
        "classification/pr_auc": float(average_precision_score(labels, probability)),
        "classification/accuracy_at_0.5": float(accuracy_score(labels, probability >= 0.5)),
        "classification/brier": float(brier_score_loss(labels, probability)),
    }
    signal = labels == 1
    energy_residual = values["energy"][signal] - values["y_energy"][signal]
    core_residual = values["core"][signal] - values["y_core"][signal]
    core_distance = np.linalg.norm(core_residual, axis=1)
    metrics.update(
        {
            "energy/mae": float(np.mean(np.abs(energy_residual))),
            "energy/rmse": float(np.sqrt(np.mean(energy_residual**2))),
            "energy/bias": float(np.mean(energy_residual)),
            "core/mean_distance": float(np.mean(core_distance)),
            "core/median_distance": float(np.median(core_distance)),
            "core/p68_distance": float(np.quantile(core_distance, 0.68)),
        }
    )
    if "direction" in values:
        cosine = np.sum(values["direction"][signal] * values["y_direction"][signal], axis=1)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        metrics["direction/median_angle_deg"] = float(np.median(angle))
        metrics["direction/p68_angle_deg"] = float(np.quantile(angle, 0.68))
    return metrics

