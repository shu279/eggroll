"""Pure-integer PyTorch EGG model and compact EGGROLL trainer."""

from .checkpoint import load_checkpoint, save_checkpoint
from .eggroll import (
    EggRollConfig,
    eggroll_step,
    evaluate_antithetic_pairs,
    make_population_evaluator,
    sample_noise,
    shape_fitness,
    update_model_,
)
from .fitness import bits_per_byte, sequence_log_likelihood_q4
from .model import DenseNoise, EggConfig, EggModel, LowRankNoise, parameter_count

__all__ = [
    "DenseNoise",
    "EggConfig",
    "EggModel",
    "EggRollConfig",
    "LowRankNoise",
    "bits_per_byte",
    "eggroll_step",
    "evaluate_antithetic_pairs",
    "load_checkpoint",
    "make_population_evaluator",
    "parameter_count",
    "sample_noise",
    "save_checkpoint",
    "sequence_log_likelihood_q4",
    "shape_fitness",
    "update_model_",
]
