import torch

from egg.checkpoint import load_checkpoint, save_checkpoint
from egg.model import EggConfig, EggModel


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(4)
    model = EggModel(EggConfig(hidden_size=16, layers=1))
    path = save_checkpoint(tmp_path / "egg.pt", model)
    restored = load_checkpoint(path)
    assert restored.config == model.config
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
