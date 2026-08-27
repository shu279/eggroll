"""Throughput benchmark for PyTorch EGG population evaluation."""

from __future__ import annotations

import argparse
import time

import torch

from .eggroll import make_population_evaluator, sample_noise
from .model import EggConfig, EggModel, parameter_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark EGGROLL population evaluation throughput"
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma-shift", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use TorchInductor; disable with --no-compile",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.population < 2 or args.population % 2:
        raise ValueError("population must be an even integer >= 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this benchmark on a CUDA VM or pass --device cpu for a smoke test."
        )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = EggModel(
        EggConfig(hidden_size=args.hidden_size, layers=args.layers)
    ).to(device)
    pair_count = args.population // 2
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    noise = sample_noise(model, generator, pair_count, args.rank)
    tokens = torch.randint(
        0,
        256,
        (pair_count, args.sequence_length + 1),
        dtype=torch.uint8,
        device=device,
    )
    evaluate = make_population_evaluator(
        model,
        sigma_shift=args.sigma_shift,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
    )

    for _ in range(args.warmup):
        evaluate(noise, tokens)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    scores = None
    for _ in range(args.iterations):
        scores = evaluate(noise, tokens)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    assert scores is not None
    checksum = int(scores.sum().item())

    evaluated_tokens = args.population * args.sequence_length * args.iterations
    tokens_per_second = evaluated_tokens / elapsed
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    print(f"device={device_name}")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"compile={args.compile} mode={args.compile_mode}"
    )
    print(
        f"model={args.layers}L-{args.hidden_size}D parameters={parameter_count(model):,} "
        f"population={args.population:,} rank={args.rank} sequence={args.sequence_length}"
    )
    print(
        f"elapsed={elapsed:.3f}s throughput={tokens_per_second:,.0f} tokens/s "
        f"peak_cuda_memory={peak_gib:.2f} GiB checksum={checksum}"
    )


if __name__ == "__main__":
    main()
