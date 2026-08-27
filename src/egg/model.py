"""PyTorch implementation of the Evolved Generative GRU (EGG).

The architecture follows Appendix G of arXiv:2511.16652. Weights and
activations stay on the symmetric int8 lattice [-127, 127]; matrix products
accumulate in int32. EGGROLL noise can be applied in low-rank form without
materialising a perturbed weight matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor, nn


I8_MIN = -127
I8_MAX = 127
FIXED_POINT_BITS = 4
FIXED_POINT_SCALE = 1 << FIXED_POINT_BITS


@dataclass
class LowRankNoise:
    """Rank-r matrix noise, represented as E = A B^T."""

    a: Tensor
    b: Tensor


@dataclass
class DenseNoise:
    """Full noise used for one-dimensional parameters."""

    value: Tensor


Noise: TypeAlias = LowRankNoise | DenseNoise
NoiseTree: TypeAlias = dict[str, Noise]


@dataclass(frozen=True)
class EggConfig:
    """EGG architecture configuration.

    The paper uses ``hidden_size=256`` and ``layers=6``. Smaller defaults make
    it possible to inspect every operation and run EGGROLL on a CPU.
    """

    vocab_size: int = 256
    hidden_size: int = 16
    layers: int = 1
    expansion: int = 4

    def __post_init__(self) -> None:
        if not 2 <= self.vocab_size <= 256:
            raise ValueError("vocab_size must be between 2 and 256")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if math.isqrt(self.hidden_size) ** 2 != self.hidden_size:
            raise ValueError(
                "hidden_size must be a perfect square (for example 16, 64, or 256)"
            )
        if self.hidden_size % 16 != 0:
            raise ValueError("hidden_size must be divisible by 16 for INT8 GEMM alignment")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.expansion != 4:
            raise ValueError("the paper's EGG architecture uses expansion=4")


def clip_i8(value: Tensor) -> Tensor:
    """Saturate to the paper's symmetric signed-int8 range."""

    return value.clamp(I8_MIN, I8_MAX).to(torch.int8)


def clipped_add(*values: Tensor) -> Tensor:
    """Saturating addition, one of EGG's implicit nonlinearities."""

    total = torch.zeros_like(values[0], dtype=torch.int32)
    for value in values:
        total = total + value.to(torch.int32)
    return clip_i8(total)


def _random_i8(shape: tuple[int, ...], device: torch.device | None = None) -> Tensor:
    values = torch.randn(shape, device=device).mul(FIXED_POINT_SCALE).round()
    return clip_i8(values)


def _int8_parameter(
    shape: tuple[int, ...], *, fill: int | None = None
) -> nn.Parameter:
    value = (
        torch.full(shape, fill, dtype=torch.int8)
        if fill is not None
        else _random_i8(shape)
    )
    # Integer tensors cannot participate in autograd. EGGROLL updates them
    # directly, but Parameter keeps familiar state_dict/named_parameters APIs.
    return nn.Parameter(value, requires_grad=False)


def _shift_noise(value: Tensor, sign: int | Tensor, sigma_shift: int) -> Tensor:
    sign_tensor = torch.as_tensor(sign, dtype=torch.int32, device=value.device)
    while sign_tensor.ndim < value.ndim:
        sign_tensor = sign_tensor.unsqueeze(-1)
    return torch.bitwise_right_shift(
        value.to(torch.int32) * sign_tensor, FIXED_POINT_BITS + sigma_shift
    )


def noisy_vector(
    value: Tensor,
    noise: DenseNoise | None,
    sign: int | Tensor,
    sigma_shift: int,
) -> Tensor:
    if noise is None:
        return value
    return clip_i8(value.to(torch.int32) + _shift_noise(noise.value, sign, sigma_shift))


def int8_mm_accumulate(x: Tensor, weight: Tensor) -> Tensor:
    """INT8 GEMM with INT32 accumulation.

    ``torch.mm`` returns int8 for int8 inputs and can overflow. PyTorch's
    ``torch._int_mm`` is the accumulating kernel lowered by Inductor to ATen,
    Triton, or CUTLASS on CUDA. The right-hand side is intentionally a
    column-major transpose view; making it contiguous is slower on CUDA.
    """

    if x.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("int8_mm_accumulate expects int8 inputs")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("input and weight widths must match")

    leading_shape = x.shape[:-1]
    flat_x = x.reshape(-1, x.shape[-1]).contiguous()
    weight_t = weight.transpose(0, 1)
    if hasattr(torch, "_int_mm") and flat_x.device.type in ("cpu", "cuda"):
        output = torch._int_mm(flat_x, weight_t)
    else:
        # Portability fallback for backends such as MPS. This path is correct,
        # but does not claim Tensor Core throughput.
        output = flat_x.to(torch.int32) @ weight_t.to(torch.int32)
    return output.reshape(*leading_shape, weight.shape[0])


def scaled_mm(
    x: Tensor,
    weight: Tensor,
    noise: LowRankNoise | None = None,
    sign: int | Tensor = 1,
    sigma_shift: int = 4,
) -> Tensor:
    """Compute ``x @ weight.T`` using EGG's fixed-point integer scaling.

    Leading dimensions are supported. For a population batch, ``x`` is
    ``[population, in]``, A is ``[population, out, rank]``, and B is
    ``[population, in, rank]``.
    """

    in_size = weight.shape[-1]
    root = math.isqrt(in_size)
    if root * root != in_size:
        raise ValueError(f"matrix input width {in_size} must be a perfect square")

    result = int8_mm_accumulate(x, weight)
    if noise is not None:
        # The unique low-rank term is bandwidth-cheap compared with the shared
        # base GEMM. Elementwise products plus reductions avoid a batched dense
        # matrix multiply for every population member.
        projected = (
            x.unsqueeze(-1).to(torch.int32) * noise.b.to(torch.int32)
        ).sum(dim=-2, dtype=torch.int32)
        low_rank = (
            projected.unsqueeze(-2) * noise.a.to(torch.int32)
        ).sum(dim=-1, dtype=torch.int32)
        result = result + _shift_noise(low_rank, sign, sigma_shift)

    scaled = torch.div(result, FIXED_POINT_SCALE * root, rounding_mode="floor")
    return clip_i8(scaled)


def embedding_lookup(
    token: Tensor,
    weight: Tensor,
    noise: LowRankNoise | None = None,
    sign: int | Tensor = 1,
    sigma_shift: int = 4,
) -> Tensor:
    token = token.to(torch.long)
    result = weight[token].to(torch.int32)
    if noise is not None:
        if token.ndim == 0:
            selected_a = noise.a[token]
        elif noise.a.ndim == token.ndim + 2 and noise.a.shape[0] == token.shape[0]:
            batch_index = torch.arange(token.shape[0], device=token.device)
            selected_a = noise.a[batch_index, token]
        else:
            selected_a = noise.a[token]
        low_rank_row = (
            selected_a.to(torch.int32).unsqueeze(-2) * noise.b.to(torch.int32)
        ).sum(dim=-1, dtype=torch.int32)
        result = result + _shift_noise(low_rank_row, sign, sigma_shift)
    return clip_i8(result)


def layer_norm_l1(
    x: Tensor,
    weight: Tensor,
    noise: DenseNoise | None = None,
    sign: int | Tensor = 1,
    sigma_shift: int = 4,
) -> Tensor:
    """Integer mean-absolute-value normalisation from Appendix G.6."""

    actual_weight = noisy_vector(weight, noise, sign, sigma_shift)
    mean_abs = (
        x.to(torch.int32).abs().sum(dim=-1, keepdim=True) // x.shape[-1]
    ).clamp_min(1)
    numerator = x.to(torch.int32) * actual_weight.to(torch.int32)
    return clip_i8(torch.div(numerator, mean_abs, rounding_mode="floor"))


def _noise(noise: NoiseTree | None, name: str) -> Noise | None:
    return None if noise is None else noise.get(name)


class Int8Linear(nn.Module):
    """A bias-free integer linear layer with optional low-rank ES noise."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _int8_parameter((out_features, in_features))

    def forward(
        self,
        x: Tensor,
        noise: LowRankNoise | None = None,
        sign: int | Tensor = 1,
        sigma_shift: int = 4,
    ) -> Tensor:
        return scaled_mm(x, self.weight, noise, sign, sigma_shift)


class Int8L1Norm(nn.Module):
    """Learnable integer L1 normalisation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = _int8_parameter((hidden_size,), fill=FIXED_POINT_SCALE)

    def forward(
        self,
        x: Tensor,
        noise: DenseNoise | None = None,
        sign: int | Tensor = 1,
        sigma_shift: int = 4,
    ) -> Tensor:
        return layer_norm_l1(x, self.weight, noise, sign, sigma_shift)


class EggGRU(nn.Module):
    """Activation-free integer minGRU variant from Appendix G.8."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.wf = Int8Linear(hidden_size, hidden_size)
        self.uf = Int8Linear(hidden_size, hidden_size)
        self.bf = _int8_parameter((hidden_size,), fill=0)
        self.wh = Int8Linear(hidden_size, hidden_size)
        self.uh = Int8Linear(hidden_size, hidden_size)
        self.bh = _int8_parameter((hidden_size,), fill=0)

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        noise: NoiseTree | None,
        prefix: str,
        sign: int | Tensor,
        sigma_shift: int,
    ) -> tuple[Tensor, Tensor]:
        f = clipped_add(
            self.wf(x, _noise(noise, f"{prefix}.wf.weight"), sign, sigma_shift),
            self.uf(state, _noise(noise, f"{prefix}.uf.weight"), sign, sigma_shift),
            noisy_vector(
                self.bf, _noise(noise, f"{prefix}.bf"), sign, sigma_shift
            ),
        )
        gate = f.to(torch.int32) + I8_MAX
        gated_past = clip_i8(torch.bitwise_right_shift(gate * state.to(torch.int32), 8))
        candidate = clipped_add(
            self.wh(x, _noise(noise, f"{prefix}.wh.weight"), sign, sigma_shift),
            self.uh(
                gated_past,
                _noise(noise, f"{prefix}.uh.weight"),
                sign,
                sigma_shift,
            ),
            noisy_vector(
                self.bh, _noise(noise, f"{prefix}.bh"), sign, sigma_shift
            ),
        )
        delta = torch.bitwise_right_shift(
            gate * (candidate.to(torch.int32) - state.to(torch.int32)), 8
        )
        new_state = clipped_add(state, clip_i8(delta))
        return new_state, new_state


class EggLayer(nn.Module):
    def __init__(self, hidden_size: int, expansion: int):
        super().__init__()
        self.ln1 = Int8L1Norm(hidden_size)
        self.gru = EggGRU(hidden_size)
        self.ln2 = Int8L1Norm(hidden_size)
        self.mlp_in = Int8Linear(hidden_size, hidden_size * expansion)
        self.mlp_out = Int8Linear(hidden_size * expansion, hidden_size)

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        noise: NoiseTree | None,
        prefix: str,
        sign: int | Tensor,
        sigma_shift: int,
    ) -> tuple[Tensor, Tensor]:
        residual = x
        x = self.ln1(
            x, _noise(noise, f"{prefix}.ln1.weight"), sign, sigma_shift
        )
        x, state = self.gru(x, state, noise, f"{prefix}.gru", sign, sigma_shift)
        x = clipped_add(x, residual)

        residual = x
        x = self.ln2(
            x, _noise(noise, f"{prefix}.ln2.weight"), sign, sigma_shift
        )
        x = self.mlp_in(
            x, _noise(noise, f"{prefix}.mlp_in.weight"), sign, sigma_shift
        )
        x = self.mlp_out(
            x, _noise(noise, f"{prefix}.mlp_out.weight"), sign, sigma_shift
        )
        return clipped_add(x, residual), state


class EggModel(nn.Module):
    """Byte-level decoder-only EGG language model."""

    def __init__(self, config: EggConfig):
        super().__init__()
        self.config = config
        self.embedding = _int8_parameter((config.vocab_size, config.hidden_size))
        self.layers = nn.ModuleList(
            [EggLayer(config.hidden_size, config.expansion) for _ in range(config.layers)]
        )
        self.ln_out = Int8L1Norm(config.hidden_size)
        self.head = Int8Linear(config.hidden_size, config.vocab_size)

    @property
    def device(self) -> torch.device:
        return self.embedding.device

    def initial_state(self, batch_size: int | None = None) -> Tensor:
        shape = (
            (self.config.layers, self.config.hidden_size)
            if batch_size is None
            else (batch_size, self.config.layers, self.config.hidden_size)
        )
        return torch.zeros(shape, dtype=torch.int8, device=self.device)

    def forward_token(
        self,
        token: Tensor,
        state: Tensor | None = None,
        noise: NoiseTree | None = None,
        sign: int | Tensor = 1,
        sigma_shift: int = 4,
    ) -> tuple[Tensor, Tensor]:
        if state is None:
            state = self.initial_state(None if token.ndim == 0 else token.shape[0])
        x = embedding_lookup(
            token,
            self.embedding,
            _noise(noise, "embedding"),
            sign,
            sigma_shift,
        )

        next_states = []
        for index, layer in enumerate(self.layers):
            x, next_state = layer(
                x,
                state[..., index, :],
                noise,
                f"layers.{index}",
                sign,
                sigma_shift,
            )
            next_states.append(next_state)

        x = self.ln_out(
            x, _noise(noise, "ln_out.weight"), sign, sigma_shift
        )
        logits = self.head(
            x, _noise(noise, "head.weight"), sign, sigma_shift
        )
        return logits, torch.stack(next_states, dim=-2)

    def forward(
        self,
        tokens: Tensor,
        state: Tensor | None = None,
        noise: NoiseTree | None = None,
        sign: int | Tensor = 1,
        sigma_shift: int = 4,
        reset_token: int | None = 0,
    ) -> tuple[Tensor, Tensor]:
        """Run tokens shaped ``[time]`` or ``[batch, time]``."""

        if tokens.ndim not in (1, 2):
            raise ValueError("tokens must have shape [time] or [batch, time]")
        if state is None:
            state = self.initial_state(None if tokens.ndim == 1 else tokens.shape[0])

        all_logits = []
        for time_index in range(tokens.shape[-1]):
            token = tokens[..., time_index]
            if reset_token is not None:
                reset = token == reset_token
                state = torch.where(
                    reset[..., None, None], torch.zeros_like(state), state
                )
            logits, state = self.forward_token(
                token, state, noise, sign, sigma_shift
            )
            all_logits.append(logits)
        return torch.stack(all_logits, dim=-2), state


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
