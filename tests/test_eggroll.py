import torch

from egg.eggroll import (
    EggRollConfig,
    eggroll_step,
    evaluate_antithetic_pairs,
    sample_noise,
    shape_fitness,
    update_model_,
)
from egg.model import EggConfig, EggModel, LowRankNoise


def test_sample_noise_is_deterministic_and_uses_parameter_names():
    torch.manual_seed(0)
    model = EggModel(EggConfig(hidden_size=16, layers=1))
    first = sample_noise(model, torch.Generator().manual_seed(7), pair_count=3)
    second = sample_noise(model, torch.Generator().manual_seed(7), pair_count=3)
    assert first.keys() == dict(model.named_parameters()).keys()
    for name in first:
        left, right = first[name], second[name]
        if isinstance(left, LowRankNoise):
            torch.testing.assert_close(left.a, right.a)
            torch.testing.assert_close(left.b, right.b)
        else:
            torch.testing.assert_close(left.value, right.value)


def test_ternary_antithetic_fitness():
    scores = torch.tensor([[10, 3], [4, 4], [-2, 9]], dtype=torch.int32)
    torch.testing.assert_close(
        shape_fitness(scores), torch.tensor([1, 0, -1], dtype=torch.int8)
    )


def test_population_evaluation_is_batched():
    torch.manual_seed(1)
    model = EggModel(EggConfig(hidden_size=16, layers=1))
    noise = sample_noise(model, torch.Generator().manual_seed(2), pair_count=2)
    tokens = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=torch.uint8)
    scores = evaluate_antithetic_pairs(model, noise, tokens)
    assert scores.shape == (2, 2)
    assert scores.dtype == torch.int32


def test_update_moves_each_weight_by_at_most_one_bin():
    torch.manual_seed(3)
    model = EggModel(EggConfig(hidden_size=16, layers=1))
    noise = sample_noise(model, torch.Generator().manual_seed(4), pair_count=2)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    update_model_(model, noise, torch.tensor([1, -1], dtype=torch.int8), alpha=1.0)

    total_changes = 0
    for name, parameter in model.named_parameters():
        difference = parameter.to(torch.int16) - before[name].to(torch.int16)
        assert difference.abs().max().item() <= 1
        total_changes += (difference != 0).sum().item()
    assert total_changes > 0


def test_streaming_step_chunks_population_and_updates_once():
    torch.manual_seed(5)
    model = EggModel(EggConfig(hidden_size=16, layers=1))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    config = EggRollConfig(
        population_size=8, pair_chunk_size=2, rank=1, alpha=1.0
    )
    tokens = torch.tensor(
        [[0, 1, 2, 3], [0, 3, 2, 1], [0, 4, 5, 6], [0, 6, 5, 4]],
        dtype=torch.uint8,
    )
    scores = eggroll_step(
        model, torch.Generator().manual_seed(6), tokens, config
    )
    assert scores.shape == (4, 2)
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
