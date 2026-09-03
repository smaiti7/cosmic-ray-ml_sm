from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalEncoder(nn.Module):
    """Shared 1D CNN applied independently to every detector waveform."""

    def __init__(self, embedding_dim: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        batch, detectors, time_bins = waveforms.shape
        x = waveforms.reshape(batch * detectors, 1, time_bins)
        x = self.network(x)
        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1)), dim=-1)
        return self.projection(pooled).reshape(batch, detectors, -1)


class BaseTemporalDeepSet(nn.Module):
    """Shared detector encoders and multi-task heads for both pooling variants."""

    model_type = "base"

    def __init__(
        self,
        positions: torch.Tensor,
        temporal_dim: int = 48,
        detector_dim: int = 64,
        event_dim: int = 128,
        predict_direction: bool = False,
    ) -> None:
        super().__init__()
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"Expected positions shaped (Ndet, 2), got {positions.shape}")
        self.register_buffer("positions", positions.float())
        self.predict_direction = predict_direction
        self.temporal_encoder = TemporalEncoder(temporal_dim)
        self.detector_encoder = nn.Sequential(
            nn.Linear(temporal_dim + 2, detector_dim),
            nn.LayerNorm(detector_dim),
            nn.GELU(),
            nn.Linear(detector_dim, detector_dim),
            nn.GELU(),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(detector_dim * 3 + 1, event_dim),
            nn.LayerNorm(event_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(event_dim, event_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(event_dim, 1)
        self.energy_head = nn.Linear(event_dim, 1)
        self.core_head = nn.Linear(event_dim, 2)
        self.direction_head = nn.Linear(event_dim, 3) if predict_direction else None

    def detector_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        batch, detectors, _ = waveforms.shape
        if detectors != self.positions.shape[0]:
            raise ValueError(f"Model has {self.positions.shape[0]} detectors, input has {detectors}")
        temporal = self.temporal_encoder(waveforms)
        positions = self.positions.unsqueeze(0).expand(batch, -1, -1)
        return self.detector_encoder(torch.cat((temporal, positions), dim=-1))

    def aggregate(self, detector_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, waveforms: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        event = self.event_encoder(self.aggregate(self.detector_features(waveforms), mask))
        output = {
            "class_logit": self.classifier(event).squeeze(-1),
            "energy": self.energy_head(event).squeeze(-1),
            "core": self.core_head(event),
        }
        if self.direction_head is not None:
            output["direction"] = F.normalize(self.direction_head(event), dim=-1)
        return output


class TemporalDeepSet(BaseTemporalDeepSet):
    """Original Deep Sets: equal-weight masked mean/std/max pooling."""

    model_type = "original"

    def aggregate(self, detector_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active = mask.unsqueeze(-1)
        features = detector_features * active
        count = active.sum(dim=1).clamp_min(1.0)
        mean = features.sum(dim=1) / count
        variance = ((features - mean.unsqueeze(1)) ** 2 * active).sum(dim=1) / count
        maximum = features.masked_fill(active == 0, torch.finfo(features.dtype).min).amax(dim=1)
        active_fraction = mask.mean(dim=1, keepdim=True)
        return torch.cat((mean, variance.clamp_min(0).sqrt(), maximum, active_fraction), dim=-1)


class AttentionTemporalDeepSet(BaseTemporalDeepSet):
    """Attention Deep Sets: learned active-detector weights for mean/std pooling."""

    model_type = "attention"

    def __init__(self, positions: torch.Tensor, **kwargs) -> None:
        super().__init__(positions, **kwargs)
        detector_dim = self.detector_encoder[0].out_features
        self.attention = nn.Sequential(
            nn.Linear(detector_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def aggregate(self, detector_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active = mask.unsqueeze(-1)
        scores = self.attention(detector_features).masked_fill(active == 0, -1e4)
        weights = torch.softmax(scores, dim=1) * active
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mean = (weights * detector_features).sum(dim=1)
        variance = (weights * (detector_features - mean.unsqueeze(1)) ** 2).sum(dim=1)
        maximum = detector_features.masked_fill(active == 0, torch.finfo(detector_features.dtype).min).amax(dim=1)
        active_fraction = mask.mean(dim=1, keepdim=True)
        return torch.cat((mean, variance.clamp_min(0).sqrt(), maximum, active_fraction), dim=-1)


def build_model(model_type: str, positions: torch.Tensor, **kwargs) -> BaseTemporalDeepSet:
    model_classes = {
        "original": TemporalDeepSet,
        "attention": AttentionTemporalDeepSet,
    }
    try:
        model_class = model_classes[model_type]
    except KeyError as error:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {sorted(model_classes)}") from error
    return model_class(positions, **kwargs)


def model_config(model: BaseTemporalDeepSet) -> dict[str, int | bool | str]:
    first_detector_layer = model.detector_encoder[0]
    first_event_layer = model.event_encoder[0]
    return {
        "model_type": model.model_type,
        "temporal_dim": int(first_detector_layer.in_features - 2),
        "detector_dim": int(first_detector_layer.out_features),
        "event_dim": int(first_event_layer.out_features),
        "predict_direction": bool(model.predict_direction),
    }
