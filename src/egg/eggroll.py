"""PyTorch implementation of integer EGGROLL for the EGG model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from scipy.stats import norm
from torch import Tensor

from .fitness import sequence_log_likelihood_q4
from .model import (
    DenseNoise,
    EggModel,
    FIXED_POINT_SCALE,
    I8_MAX,
    I8_MIN,
    LowRankNoise,
    NoiseTree,
)


@dataclass(frozen=True)
class EggRollConfig:
    population_size: int = 32
    rank: int = 1
    sigma_shift: int = 4
    alpha: float = 0.1
    pair_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("population_size must be an even integer >= 2")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.sigma_shift < 0:
            raise ValueError("sigma_shift must be non-negative")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.pair_chunk_size is not None and self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if (
            self.pair_chunk_size is not None
            and self.pair_count % self.pair_chunk_size != 0
        ):
            raise ValueError("pair_chunk_size must divide the antithetic pair count")

    @property
    def pair_count(self) -> int:
        return self.population_size // 2


def _normal_i8(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> Tensor:
    return (
        torch.randn(shape, generator=generator, device=device)
        .mul(FIXED_POINT_SCALE)
        .round()
        .clamp(I8_MIN, I8_MAX)
        .to(torch.int8)
    )


@torch.no_grad()
def sample_noise(
    model: EggModel,
    generator: torch.Generator,
    pair_count: int,
    rank: int = 1,
) -> NoiseTree:
    """Sample Q4 noise keyed by ``model.named_parameters()`` names."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    noise: NoiseTree = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2:
            out_size, in_size = parameter.shape
            noise[name] = LowRankNoise(
                _normal_i8(
                    (pair_count, out_size, rank), generator, parameter.device
                ),
                _normal_i8(
                    (pair_count, in_size, rank), generator, parameter.device
                ),
            )
        elif parameter.ndim == 1:
            left = _normal_i8(
                (pair_count,) + tuple(parameter.shape), generator, parameter.device
            ).to(torch.int32)
            right = _normal_i8(
                (pair_count,) + tuple(parameter.shape), generator, parameter.device
            ).to(torch.int32)
            noise[name] = DenseNoise(left * right)
        else:
            raise ValueError(f"unsupported parameter rank for {name}: {parameter.ndim}")
    return noise


@torch.no_grad()
def evaluate_antithetic_pairs(
    model: EggModel,
    noise: NoiseTree,
    token_batches: Tensor,
    sigma_shift: int = 4,
) -> Tensor:
    """Evaluate all pairs together; return ``[pairs, (+, -)]`` Q4 scores."""

    positive = sequence_log_likelihood_q4(
        model, token_batches, noise, sign=1, sigma_shift=sigma_shift
    )
    negative = sequence_log_likelihood_q4(
        model, token_batches, noise, sign=-1, sigma_shift=sigma_shift
    )
    return torch.stack((positive, negative), dim=-1)


PopulationEvaluator = Callable[[NoiseTree, Tensor], Tensor]


def make_population_evaluator(
    model: EggModel,
    sigma_shift: int = 4,
    compile_model: bool = False,
    compile_mode: str = "max-autotune",
) -> PopulationEvaluator:
    """Create an eager or TorchInductor population evaluator.

    Static population chunk and sequence shapes allow CUDA graph capture and
    kernel autotuning. The first compiled call is intentionally much slower.
    """

    def evaluate(noise: NoiseTree, token_batches: Tensor) -> Tensor:
        return evaluate_antithetic_pairs(model, noise, token_batches, sigma_shift)

    if not compile_model:
        return evaluate
    return torch.compile(evaluate, dynamic=False, mode=compile_mode)


def shape_fitness(pair_scores: Tensor) -> Tensor:
    """Appendix H.2 ternary fitness: sign(score+ - score-)."""

    if pair_scores.ndim != 2 or pair_scores.shape[1] != 2:
        raise ValueError("pair_scores must have shape [pair_count, 2]")
    return torch.sign(pair_scores[:, 0] - pair_scores[:, 1]).to(torch.int8)


def _matrix_direction(noise: LowRankNoise, fitness: Tensor) -> Tensor:
    # Sum_p fitness[p] * A[p] @ B[p].T as one integer matrix product.
    weighted_a = noise.a.to(torch.int32) * fitness.to(torch.int32)[:, None, None]
    out_size = weighted_a.shape[1]
    in_size = noise.b.shape[1]
    left = weighted_a.permute(1, 0, 2).reshape(out_size, -1)
    right = noise.b.to(torch.int32).permute(0, 2, 1).reshape(-1, in_size)
    return left @ right


def _dense_direction(noise: DenseNoise, fitness: Tensor) -> Tensor:
    view_shape = (fitness.shape[0],) + (1,) * (noise.value.ndim - 1)
    return (
        fitness.to(torch.int32).reshape(view_shape) * noise.value.to(torch.int32)
    ).sum(dim=0, dtype=torch.int32)


def update_threshold(alpha: float, pair_count: int) -> int:
    """Integer Z-test threshold from Appendix H.3."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return int(round(norm.ppf(1.0 - alpha / 2.0) * 256.0 * math.sqrt(pair_count)))


@torch.no_grad()
def init_directions(model: EggModel) -> dict[str, Tensor]:
    """Allocate one int32 fused-update tensor per model parameter."""

    return {
        name: torch.zeros_like(parameter, dtype=torch.int32)
        for name, parameter in model.named_parameters()
    }


@torch.no_grad()
def accumulate_directions_(
    directions: dict[str, Tensor],
    noise: NoiseTree,
    fitness: Tensor,
) -> dict[str, Tensor]:
    """Fuse one population chunk into persistent full-rank directions."""

    for name, one_noise in noise.items():
        chunk_direction = (
            _matrix_direction(one_noise, fitness)
            if isinstance(one_noise, LowRankNoise)
            else _dense_direction(one_noise, fitness)
        )
        directions[name].add_(chunk_direction)
    return directions


@torch.no_grad()
def apply_directions_(
    model: EggModel,
    directions: dict[str, Tensor],
    alpha: float,
    pair_count: int,
) -> EggModel:
    """Apply a fused direction to int8 parameters by at most one bin."""

    threshold = update_threshold(alpha, pair_count)
    for name, parameter in model.named_parameters():
        direction = directions[name]
        movement = torch.where(
            direction.abs() >= threshold,
            direction.sign(),
            torch.zeros_like(direction),
        )
        parameter.copy_(
            (parameter.to(torch.int32) + movement)
            .clamp(I8_MIN, I8_MAX)
            .to(torch.int8)
        )
    return model


@torch.no_grad()
def update_model_(
    model: EggModel,
    noise: NoiseTree,
    fitness: Tensor,
    alpha: float = 0.1,
) -> EggModel:
    """One-shot in-place update kept as a convenient small-population API."""

    directions = init_directions(model)
    accumulate_directions_(directions, noise, fitness)
    return apply_directions_(model, directions, alpha, int(fitness.shape[0]))


@torch.no_grad()
def eggroll_step(
    model: EggModel,
    generator: torch.Generator,
    token_batches: Tensor,
    config: EggRollConfig,
    evaluator: PopulationEvaluator | None = None,
) -> Tensor:
    """Evaluate and update one generation with bounded GPU memory.

    ``token_batches`` contains one sequence per antithetic pair. Noise is
    generated and discarded chunk-by-chunk; only full-size int32 update
    directions persist. This makes very large populations possible without
    storing all A/B factors at once.
    """

    if token_batches.ndim != 2:
        raise ValueError("token_batches must have shape [pair_count, time]")
    if token_batches.shape[0] != config.pair_count:
        raise ValueError(
            f"expected {config.pair_count} pairs, got {token_batches.shape[0]}"
        )
    if evaluator is None:
        evaluator = make_population_evaluator(model, config.sigma_shift)

    chunk_size = config.pair_chunk_size or config.pair_count
    directions = init_directions(model)
    all_scores = []
    for start in range(0, config.pair_count, chunk_size):
        stop = min(start + chunk_size, config.pair_count)
        noise = sample_noise(model, generator, stop - start, config.rank)
        scores = evaluator(noise, token_batches[start:stop])
        fitness = shape_fitness(scores)
        accumulate_directions_(directions, noise, fitness)
        all_scores.append(scores)

    apply_directions_(model, directions, config.alpha, config.pair_count)
    return torch.cat(all_scores, dim=0)
