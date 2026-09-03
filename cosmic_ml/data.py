from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_DATA_DIR = Path("data")


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress the long-tailed signal while preserving negative noise values."""
    return np.sign(values) * np.log1p(np.abs(values))


def load_stats(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    path = Path(data_dir) / "processed" / "stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m cosmic_ml.prepare_data")
    with path.open() as handle:
        return json.load(handle)


def load_splits(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, np.ndarray]:
    path = Path(data_dir) / "processed" / "splits.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m cosmic_ml.prepare_data")
    with np.load(path) as archive:
        return {name: archive[name].astype(np.int64) for name in archive.files}


class CosmicDataset(Dataset):
    """Memory-mapped event dataset.

    Waveforms are returned as ``(Ndet, T)`` tensors. Regression labels that are
    undefined for background events are replaced by zeros; the training loss
    uses ``y_class`` to mask them out.
    """

    def __init__(
        self,
        indices: Sequence[int] | np.ndarray,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        extracted = self.data_dir / "extracted"
        self.X = np.load(extracted / "X.npy", mmap_mode="r")
        self.masks = np.load(extracted / "detector_masks.npy", mmap_mode="r")
        self.y_class = np.load(extracted / "y_class.npy", mmap_mode="r")
        self.y_energy = np.load(extracted / "y_energy.npy", mmap_mode="r")
        self.y_core = np.load(extracted / "y_core.npy", mmap_mode="r")
        self.y_direction = np.load(extracted / "y_direction.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.stats = stats or load_stats(self.data_dir)

        self.input_mean = np.float32(self.stats["input"]["mean"])
        self.input_std = np.float32(self.stats["input"]["std"])
        self.energy_mean = np.float32(self.stats["energy"]["mean"])
        self.energy_std = np.float32(self.stats["energy"]["std"])
        self.core_mean = np.asarray(self.stats["core"]["mean"], dtype=np.float32)
        self.core_std = np.asarray(self.stats["core"]["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        event_index = int(self.indices[item])
        # np.array makes writable, contiguous arrays for zero-copy torch conversion.
        waveform = np.array(self.X[event_index], dtype=np.float32, copy=True)
        mask = np.array(self.masks[event_index], dtype=np.float32, copy=True)
        waveform = signed_log1p(waveform)
        waveform = ((waveform - self.input_mean) / self.input_std) * mask[None, :]
        waveform = np.ascontiguousarray(waveform.T)

        label = np.float32(self.y_class[event_index])
        if label == 1:
            energy = (np.float32(self.y_energy[event_index]) - self.energy_mean) / self.energy_std
            core = (np.array(self.y_core[event_index], dtype=np.float32) - self.core_mean) / self.core_std
            direction = np.array(self.y_direction[event_index], dtype=np.float32)
        else:
            energy = np.float32(0.0)
            core = np.zeros(2, dtype=np.float32)
            direction = np.zeros(3, dtype=np.float32)

        return {
            "waveform": torch.from_numpy(waveform),
            "mask": torch.from_numpy(mask),
            "y_class": torch.tensor(label),
            "y_energy": torch.tensor(energy),
            "y_core": torch.from_numpy(np.asarray(core, dtype=np.float32)),
            "y_direction": torch.from_numpy(direction),
            "index": torch.tensor(event_index, dtype=torch.int64),
        }


def normalized_positions(
    data_dir: str | Path, stats: dict[str, Any] | None = None
) -> np.ndarray:
    data_dir = Path(data_dir)
    stats = stats or load_stats(data_dir)
    positions = np.load(data_dir / "extracted" / "positions.npy").astype(np.float32)
    mean = np.asarray(stats["positions"]["mean"], dtype=np.float32)
    std = np.asarray(stats["positions"]["std"], dtype=np.float32)
    return (positions - mean) / std

