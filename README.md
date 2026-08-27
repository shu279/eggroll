# EGG model - PyTorch で学ぶ小さな再現版

論文 [Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652) の
Appendix G/H に基づく、Evolved Generative GRU（EGG）と整数版 EGGROLL の
PyTorch 実装です。

公式の [nano-egg](https://github.com/ESHyperscale/nano-egg) は JAX と H100、巨大な
population を前提にしています。この版は PyTorch の `nn.Module`、`Parameter`、
`state_dict`、batched Tensor 演算を読みながら、CPU でもアルゴリズムを確認できることを
優先しています。

## EGG と EGGROLL

- **EGG**: weight と activation が `torch.int8` の recurrent language model
- **EGGROLL**: backpropagation の代わりに低ランクのランダム摂動で EGG を学習する方法

EGG は activation 関数を使いません。int8 へ戻すときの clipping と飽和加算が
非線形性になります。行列積は `torch.int32` で accumulate してから int8 に戻します。

EGGROLL では個々の摂動を `E = A @ B.T` とします。forward は重い E を作らず、
次のように計算します。

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
`requires_grad=False` です。EGGROLL が `torch.inference_mode()` 内で parameter を直接更新します。

## 小さな学習を動かす

組み込み byte corpus:

```bash
egg-train --steps 3 --population 8 --hidden-size 16 --layers 1
```

手元のテキスト:

```bash
egg-train \
  --data ./corpus.txt \
  --steps 100 \
  --population 256 \
  --sequence-length 100 \
  --hidden-size 64 \
  --layers 2 \
  --checkpoint checkpoints/egg.pt
```

CUDA 環境では `--device cuda` を追加できます。Apple Silicon でも correctness-first の
CPU 版として動かせます。

## PyTorch として見る場所

| 学びたいこと | ファイル / class |
|---|---|
| `nn.Module` の組み立て | `egg.model.EggModel`, `EggLayer`, `EggGRU` |
| 独自整数 Linear | `egg.model.Int8Linear`, `scaled_mm` |
| recurrent state | `EggModel.forward_token`, `EggModel.forward` |
| batched population | `egg.eggroll.evaluate_antithetic_pairs` |
| `named_parameters()` の利用 | `sample_noise`, `update_model_` |
| checkpoint | `egg.checkpoint` の `state_dict` 保存・読込 |

論文との対応:

| 論文 | 実装 |
|---|---|
| Appendix G.4 scaled integer matmul | `egg.model.scaled_mm` |
| Appendix G.6 L1 LayerNorm | `egg.model.layer_norm_l1` |
| Appendix G.8 modified minGRU | `egg.model.EggGRU` |
| Appendix G.9 integer fitness | `egg.fitness` |
| Appendix H.1 rank-r perturbation | `egg.eggroll.sample_noise` |
| Appendix H.2 fitness shaping | `egg.eggroll.shape_fitness` |
| Appendix H.3 discrete update | `egg.eggroll.update_model_` |

## 再現範囲

論文構成は `hidden_size=256`, `layers=6`, population 最大 `2^20` で、モデルは
4,856,064 parameters です。論文の 3.40 bits/byte を再現するには H100 級の計算資源と
MiniPile が必要です。

この実装は数式と dtype の意味を再現していますが、公式の fused JAX/CUDA kernel は
移植していません。そのため PyTorch の通常 Tensor 演算として読みやすい一方、論文と同じ
スループットを保証するものではありません。
