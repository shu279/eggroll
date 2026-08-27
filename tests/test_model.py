import numpy as np
import pytest
import torch

from egg.fitness import fixed_log2_q4, token_log_likelihood_q4
from egg.model import (
    EggConfig,
    EggModel,
    int8_mm_accumulate,
    layer_norm_l1,
    parameter_count,
)


def test_parameter_count_matches_paper_formula():
    torch.manual_seed(0)
    config = EggConfig(vocab_size=256, hidden_size=16, layers=2)
    model = EggModel(config)
    d = config.hidden_size
    expected = 513 * d + config.layers * (4 * d + 12 * d * d)
    assert parameter_count(model) == expected
    assert all(parameter.dtype == torch.int8 for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_hidden_size_requires_integer_gemm_alignment():
    with pytest.raises(ValueError, match="divisible by 16"):
        EggConfig(hidden_size=25)


def test_int8_mm_uses_int32_accumulation_without_overflow():
    x = torch.full((3, 16), 127, dtype=torch.int8)
    weight = torch.full((5, 16), 127, dtype=torch.int8)
    actual = int8_mm_accumulate(x, weight)
    expected = torch.full((3, 5), 127 * 127 * 16, dtype=torch.int32)
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual, expected)


def test_forward_shapes_dtype_and_range():
    torch.manual_seed(1)
    model = EggModel(EggConfig(vocab_size=256, hidden_size=16, layers=2))
    tokens = torch.tensor([0, 101, 103, 103], dtype=torch.uint8)
    logits, state = model(tokens)
    assert logits.shape == (4, 256)
    assert state.shape == (2, 16)
    assert logits.dtype == torch.int8
    assert state.dtype == torch.int8
    assert logits.min().item() >= -127
    assert logits.max().item() <= 127


def test_batched_forward_uses_pytorch_batch_dimension():
    model = EggModel(EggConfig(hidden_size=16, layers=1))
    tokens = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.uint8)
    logits, state = model(tokens)
    assert logits.shape == (2, 3, 256)
    assert state.shape == (2, 1, 16)


def test_zero_layer_norm_is_safe():
    x = torch.zeros(16, dtype=torch.int8)
    weight = torch.full((16,), 16, dtype=torch.int8)
    actual = layer_norm_l1(x, weight)
    torch.testing.assert_close(actual, torch.zeros(16, dtype=torch.int8))


def test_compact_integer_log_table_is_q4_accurate():
    values = torch.tensor([16, 31, 257, 65_535, 10_000_000], dtype=torch.int32)
    actual = fixed_log2_q4(values).numpy()
    expected = np.rint(16 * np.log2(values.numpy() / 16)).astype(np.int32)
    np.testing.assert_allclose(actual, expected, atol=1)


def test_uniform_logits_give_eight_bits_per_byte():
    logits = torch.zeros(256, dtype=torch.int8)
    score = token_log_likelihood_q4(logits, torch.tensor(42, dtype=torch.uint8))
    assert abs(score.item() - (-8 * 16)) <= 1
