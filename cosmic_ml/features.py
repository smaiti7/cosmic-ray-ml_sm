from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "log_total_positive_signal",
    "log_max_sample",
    "log_max_detector_charge",
    "detectors_peak_gt_3",
    "detectors_peak_gt_5",
    "detectors_peak_gt_10",
    "active_fraction",
    "charge_centroid_x",
    "charge_centroid_y",
    "charge_radial_spread",
    "arrival_time_mean",
    "arrival_time_std",
    "arrival_gradient_x",
    "arrival_gradient_y",
    "arrival_gradient_norm",
    "arrival_plane_residual",
]


def extract_features(data_dir: str | Path, chunk_size: int = 256) -> np.ndarray:
    """Extract interpretable, event-level physics features from all events."""
    data_dir = Path(data_dir)
    extracted = data_dir / "extracted"
    X = np.load(extracted / "X.npy", mmap_mode="r")
    masks = np.load(extracted / "detector_masks.npy", mmap_mode="r")
    positions = np.load(extracted / "positions.npy").astype(np.float64)
    features = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float32)
    times = np.arange(X.shape[1], dtype=np.float64)

    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        values = np.asarray(X[start:stop], dtype=np.float32)
        mask = np.asarray(masks[start:stop], dtype=np.float32)
        positive = np.maximum(values, 0.0) * mask[:, None, :]
        charge = positive.sum(axis=1, dtype=np.float64)
        total = charge.sum(axis=1)
        safe_total = np.maximum(total, 1e-8)
        peak = positive.max(axis=1)
        arrival = values.argmax(axis=1).astype(np.float64)
        centroid = (charge @ positions) / safe_total[:, None]
        displacement = positions[None, :, :] - centroid[:, None, :]
        radial_spread = np.sqrt(
            (charge * np.sum(displacement**2, axis=2)).sum(axis=1) / safe_total
        )
        time_mean = (charge * arrival).sum(axis=1) / safe_total
        time_std = np.sqrt(
            (charge * (arrival - time_mean[:, None]) ** 2).sum(axis=1) / safe_total
        )

        row = features[start:stop]
        row[:, 0] = np.log1p(total)
        row[:, 1] = np.log1p(positive.max(axis=(1, 2)))
        row[:, 2] = np.log1p(charge.max(axis=1))
        row[:, 3] = (peak > 3.0).sum(axis=1)
        row[:, 4] = (peak > 5.0).sum(axis=1)
        row[:, 5] = (peak > 10.0).sum(axis=1)
        row[:, 6] = mask.mean(axis=1)
        row[:, 7:9] = centroid
        row[:, 9] = radial_spread
        row[:, 10] = time_mean
        row[:, 11] = time_std

        for local in range(stop - start):
            valid = (peak[local] > 3.0) & (mask[local] > 0.5)
            if valid.sum() < 3:
                continue
            design = np.column_stack((np.ones(valid.sum()), positions[valid]))
            weights = np.sqrt(np.maximum(charge[local, valid], 1e-6))
            weighted_design = design * weights[:, None]
            weighted_time = arrival[local, valid] * weights
            coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_time, rcond=None)
            gradient = coefficients[1:]
            residual = arrival[local, valid] - design @ coefficients
            row[local, 12:14] = gradient
            row[local, 14] = np.linalg.norm(gradient)
            row[local, 15] = np.sqrt(np.average(residual**2, weights=charge[local, valid]))
        if start == 0 or stop == len(X) or start % (20 * chunk_size) == 0:
            print(f"  feature extraction: {stop}/{len(X)}")
    return features


def load_or_extract_features(data_dir: str | Path, force: bool = False) -> np.ndarray:
    data_dir = Path(data_dir)
    path = data_dir / "processed" / "physics_features.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"Loading cached features from {path}")
        return np.load(path, mmap_mode="r")
    features = extract_features(data_dir)
    np.save(path, features)
    print(f"Saved features to {path}")
    return features

