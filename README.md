# eggroll

A compact, CUDA-first PyTorch reproduction of EGG and EGGROLL from [Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652). JAX is not required: this implementation uses PyTorch, integer GEMM, and `torch.compile` for GPU execution.

## Features

- INT8 EGG recurrent layers with INT32 accumulation
- Activation-free GRU-style updates and L1 normalization
- Antithetic low-rank EGGROLL updates
- Chunked population evaluation to control GPU memory use
- Optional `torch.compile` acceleration

## Install and test

```bash
pip install -e '.[dev]'
pytest -q
```

## Train

```bash
egg-train \
  --data corpus.txt \
  --device cuda \
  --hidden-size 256 \
  --layers 6 \
  --population 4096 \
  --pair-chunk-size 1024 \
  --sequence-length 100 \
  --steps 100
```

## Benchmark

```bash
egg-benchmark \
  --device cuda \
  --hidden-size 256 \
  --layers 6 \
  --population 4096 \
  --sequence-length 100
```

An NVIDIA H100 is recommended; on GCE, `a3-highgpu-1g` is a good starting point. The current CLI uses one GPU. Paper-scale throughput has not yet been reproduced, and `torch._int_mm` is a private PyTorch API that may change.

References: [paper](https://arxiv.org/abs/2511.16652) · [official nano-egg](https://github.com/ESHyperscale/nano-egg)
