"""Integer fixed-point next-byte fitness for the PyTorch EGG model."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .model import EggModel, NoiseTree


Q_BITS = 4
Q_SCALE = 1 << Q_BITS

_EXP2_Q4 = torch.from_numpy(
    (np.exp2(np.arange(256, dtype=np.float64) / Q_SCALE) * Q_SCALE).astype(np.int32)
)
_LOG2_MANTISSA_Q4 = torch.from_numpy(
    np.rint(
        np.log2(1.0 + np.arange(4096, dtype=np.float64) / 4096.0) * Q_SCALE
    ).astype(np.int32)
)


def fixed_log2_q4(x_q4: Tensor) -> Tensor:
    """Approximate ``16 * log2(x_q4 / 16)`` using integer tensor ops."""

    x = x_q4.to(torch.int64).clamp_min(1)
    exponent = torch.zeros_like(x, dtype=torch.int64)
    # x never exceeds 2^28 in the byte-level softmax. Comparisons keep this
    # educational version portable without relying on a device-specific CLZ op.
    for bit in range(1, 31):
        exponent = exponent + (x >= (1 << bit)).to(torch.int64)

    right = (exponent - 12).clamp_min(0)
    left = (12 - exponent).clamp_min(0)
    normalized = torch.where(
        exponent >= 12,
        torch.bitwise_right_shift(x, right),
        torch.bitwise_left_shift(x, left),
    )
    index = (normalized - 4096).clamp(0, 4095).to(torch.long)
    lut = _LOG2_MANTISSA_Q4.to(x.device)
    return ((exponent - 4) * Q_SCALE + lut[index]).to(torch.int32)


def token_log_likelihood_q4(logits: Tensor, targets: Tensor) -> Tensor:
    """Return next-byte log2 likelihood in signed Q4 fixed point."""

    shifted = logits.to(torch.int32) + 128
    targets = targets.to(torch.long)
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    exp_table = _EXP2_Q4.to(logits.device)
    exp_sum_q4 = exp_table[shifted.to(torch.long)].sum(dim=-1, dtype=torch.int32)
    return target_logits - fixed_log2_q4(exp_sum_q4)


@torch.inference_mode()
def sequence_log_likelihood_q4(
    model: EggModel,
    tokens: Tensor,
    noise: NoiseTree | None = None,
    sign: int | Tensor = 1,
    sigma_shift: int = 4,
) -> Tensor:
    """Score sequences shaped ``[time]`` or ``[batch, time]``."""

    logits, _ = model(
        tokens[..., :-1], noise=noise, sign=sign, sigma_shift=sigma_shift
    )
    scores = token_log_likelihood_q4(logits, tokens[..., 1:])
    return scores.sum(dim=-1, dtype=torch.int32)


def bits_per_byte(log_likelihood_q4: Tensor, token_count: int) -> float:
    if log_likelihood_q4.numel() != 1:
        raise ValueError("bits_per_byte expects a scalar score")
    return -float(log_likelihood_q4.item()) / (token_count * Q_SCALE)
