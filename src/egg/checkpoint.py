"""Standard PyTorch checkpoints for EGG."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .model import EggConfig, EggModel


def save_checkpoint(path: str | Path, model: EggModel) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.config), "state_dict": model.state_dict()},
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> EggModel:
    """Load a trusted checkpoint created by :func:`save_checkpoint`."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = EggModel(EggConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model
