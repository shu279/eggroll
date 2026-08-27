# EGG model

論文 [Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652) に基づく、Evolved Generative GRU（EGG）と整数版 EGGROLL のPyTorch 実装です。

公式の [nano-egg](https://github.com/ESHyperscale/nano-egg) は JAX で実装されていますが、EGGROLLがJAXを必要とするわけではありません。低ランク摂動、antithetic sampling、fitness shapingはすべてPyTorch Tensorで同じように実装できます。このリポジトリはCUDA-firstのPyTorch実装です。

- **EGG**: weight と activation が int8 の recurrent language model
- **EGGROLL**: backpropagation の代わりに低ランクのランダム摂動で EGG を学習する方法

EGG は activation 関数を使いません。int8 へ戻すときの clipping と飽和加算が
非線形性になります。行列積は `torch.int32` で accumulate してから int8 に戻します。

```python
base = x @ weight.T
perturbation = (x @ B) @ A.T
output = base + perturbation
```

個々の摂動が rank 1 でも、population 全体の outer product を合計した parameter update は
full-rank になり得ます。

## セットアップ

```bash
cd /Users/shusato/Desktop/eggroll
python -m pip install -e '.[dev]'
pytest -q
```

CUDA VMでは先に[PyTorch公式のインストール手順](https://pytorch.org/get-started/locally/)で、そのCUDA環境に合うwheelを入れてください。

## まず forward を読む

```python
import torch
from egg import EggConfig, EggModel

torch.manual_seed(0)

config = EggConfig(
    vocab_size=256,  # byte-level vocabulary
    hidden_size=16, # 論文は 256
    layers=1,       # 論文は 6
)
model = EggModel(config)

tokens = torch.tensor([0, 72, 101, 108, 108, 111], dtype=torch.uint8)
logits, state = model(tokens)

print(logits.shape)  # torch.Size([6, 256])
print(logits.dtype)  # torch.int8
print(state.shape)   # torch.Size([1, 16])
```

通常の PyTorch model と同じ `nn.Module` ですが、int8 parameter は autograd できないため
`requires_grad=False` です。EGGROLL が `torch.no_grad()` 内で parameter を直接更新します。

## CUDAで学習する

まず小さいsmoke test:

```bash
egg-train --steps 3 --population 8 --hidden-size 16 --layers 1
```

H100でpopulationを増やす例:

```bash
egg-train \
  --data ./corpus.txt \
  --steps 100 \
  --population 4096 \
  --pair-chunk-size 1024 \
  --sequence-length 100 \
  --hidden-size 256 \
  --layers 6 \
  --device cuda \
  --checkpoint checkpoints/egg.pt
```

CUDAでは`torch.compile(mode="max-autotune")`がデフォルトで有効になります。最初のstepはコンパイルとautotuningのため遅くなります。`--pair-chunk-size`は同時にGPUへ載せるantithetic pair数で、全pair数（population / 2）の約数にします。noiseはchunkごとに生成・破棄されるため、巨大populationでもVRAM使用量はchunkサイズで制御できます。

throughputとpeak VRAMを測るには:

```bash
egg-benchmark \
  --device cuda \
  --hidden-size 256 \
  --layers 6 \
  --population 4096 \
  --sequence-length 100
```

base GEMMは`torch._int_mm`を使い、int8入力をint32へaccumulateします。PyTorch InductorはCUDA上でATen、Triton、CUTLASSの候補をautotuneします。`torch._int_mm`は非公開APIなので、実験結果にはPyTorch/CUDA/GPUバージョンも一緒に記録してください。

## GCE

最初はCompute Engineの`a3-highgpu-1g`（H100 80GB）で十分です。現在このmachine typeはSpot VMまたはFlex-start VMで作成する必要があります。構成と最新の提供条件は[Google CloudのGPU machine type一覧](https://cloud.google.com/compute/docs/gpus)を確認してください。

このCLIは現在1 process / 1 GPUです。`a3-highgpu-8g`を使う前に、まず1 GPUで`egg-benchmark`を回して適切なpopulationとchunk sizeを決めるのがおすすめです。

## PyTorch として見る場所

| 学びたいこと                   | ファイル / class                                 |
| ------------------------ | -------------------------------------------- |
| `nn.Module` の組み立て        | `egg.model.EggModel`, `EggLayer`, `EggGRU`   |
| 独自整数 Linear              | `egg.model.Int8Linear`, `scaled_mm`          |
| recurrent state          | `EggModel.forward_token`, `EggModel.forward` |
| CUDA int8 GEMM           | `egg.model.int8_mm_accumulate`               |
| batched population       | `evaluate_antithetic_pairs`, `eggroll_step`  |
| `named_parameters()` の利用 | `sample_noise`, `update_model_`              |
| checkpoint               | `egg.checkpoint` の `state_dict` 保存・読込        |
| GPU benchmark            | `egg.benchmark`                              |

論文との対応:

| 論文                                 | 実装                          |
| ---------------------------------- | --------------------------- |
| Appendix G.4 scaled integer matmul | `egg.model.scaled_mm`       |
| Appendix G.6 L1 LayerNorm          | `egg.model.layer_norm_l1`   |
| Appendix G.8 modified minGRU       | `egg.model.EggGRU`          |
| Appendix G.9 integer fitness       | `egg.fitness`               |
| Appendix H.1 rank-r perturbation   | `egg.eggroll.sample_noise`  |
| Appendix H.2 fitness shaping       | `egg.eggroll.shape_fitness` |
| Appendix H.3 discrete update       | `egg.eggroll.update_model_` |

## 再現範囲

論文構成は `hidden_size=256`, `layers=6`, population 最大 `2^20` で、モデルは
4,856,064 parameters です。論文の 3.40 bits/byte を再現するには H100 級の計算資源と
MiniPile が必要です。

この実装はPyTorchのint8 accumulating GEMM、population batching、streaming update、`torch.compile`を利用します。JAX版とアルゴリズム上の制約は変わりません。ただし専用Triton kernelやmulti-GPUのfitness/direction集約はまだ実装していないため、論文と同じスループットを主張する前に`egg-benchmark`で測定してください。
