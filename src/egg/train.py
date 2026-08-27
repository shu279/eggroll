"""Small PyTorch EGGROLL training script."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .checkpoint import save_checkpoint
from .eggroll import (
    EggRollConfig,
    eggroll_step,
    make_population_evaluator,
    shape_fitness,
)
from .fitness import bits_per_byte, sequence_log_likelihood_q4
from .model import EggConfig, EggModel, parameter_count


DEFAULT_TEXT = (
    "eggroll learns without backprop. the egg remembers bytes. "
    "low rank noise makes a full rank update. "
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_bytes(path: str | None) -> np.ndarray:
    if path is None:
        raw = ("\x00" + DEFAULT_TEXT) * 128
        return np.frombuffer(raw.encode("utf-8"), dtype=np.uint8).copy()
    data = Path(path).read_bytes()
    if not data:
        raise ValueError("training file is empty")
    return np.concatenate(
        (np.zeros(1, dtype=np.uint8), np.frombuffer(data, dtype=np.uint8))
    )


def _sample_batches(
    data: np.ndarray,
    pair_count: int,
    sequence_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Tensor:
    starts = rng.integers(0, data.size, size=pair_count)
    offsets = np.arange(sequence_length + 1)
    indices = (starts[:, None] + offsets[None, :]) % data.size
    return torch.as_tensor(data[indices], dtype=torch.uint8, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small pure-int8 EGG model")
    parser.add_argument("--data", help="raw byte/text file; omitted uses a demo corpus")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--sigma-shift", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        help="antithetic pairs evaluated at once; omitted evaluates all pairs",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use torch.compile (default: enabled on CUDA)",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint", default="checkpoints/egg-small.pt")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = EggModel(
        EggConfig(hidden_size=args.hidden_size, layers=args.layers, vocab_size=256)
    ).to(device)
    roll = EggRollConfig(
        population_size=args.population,
        rank=args.rank,
        sigma_shift=args.sigma_shift,
        alpha=args.alpha,
        pair_chunk_size=args.pair_chunk_size,
    )
    data = _load_bytes(args.data)
    host_rng = np.random.default_rng(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    compile_enabled = args.compile if args.compile is not None else device.type == "cuda"
    evaluator = make_population_evaluator(
        model,
        sigma_shift=roll.sigma_shift,
        compile_model=compile_enabled,
        compile_mode=args.compile_mode,
    )

    validation = torch.as_tensor(
        data[: min(data.size, 513)], dtype=torch.uint8, device=device
    )
    initial_score = sequence_log_likelihood_q4(model, validation)
    print(
        f"device={device} parameters={parameter_count(model):,} "
        f"initial_bpb={bits_per_byte(initial_score, validation.numel() - 1):.3f}"
    )

    for step in range(args.steps):
        batches = _sample_batches(
            data, roll.pair_count, args.sequence_length, host_rng, device
        )
        pair_scores = eggroll_step(
            model, noise_generator, batches, roll, evaluator
        )
        fitness = shape_fitness(pair_scores)

        mean_bpb = -pair_scores.to(torch.float32).mean().item() / (
            args.sequence_length * 16.0
        )
        decisive = (fitness != 0).to(torch.float32).mean().item()
        print(
            f"step={step + 1:04d} population_bpb={mean_bpb:.3f} "
            f"decisive_pairs={decisive:.1%}"
        )

    final_score = sequence_log_likelihood_q4(model, validation)
    destination = save_checkpoint(args.checkpoint, model)
    print(
        f"final_bpb={bits_per_byte(final_score, validation.numel() - 1):.3f} "
        f"checkpoint={destination}"
    )


if __name__ == "__main__":
    main()
