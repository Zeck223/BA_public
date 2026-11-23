"""Neural-network workflows built on the shared project infrastructure."""

from typing import Any

from .data import SimDataset, SimulationDataService
from .data.schema import (
    display_parameter_name,
    get_simulation_parameter_names,
    get_target_columns,
)

__all__ = [
    "SimulationDataService",
    "SimDataset",
    "Evaluator",
    "ModelCheckpoint",
    "Trainer",
    "NNLogger",
    "display_parameter_name",
    "get_simulation_parameter_names",
    "get_target_columns",
]


def __getattr__(name: str) -> Any:
    """Resolve model and training APIs lazily.

    Args:
        name: Public neural-network attribute being requested.

    Returns:
        Any: The matching class.

    Raise:
        AttributeError: If the requested name is not exported.
    """
    if name == "Evaluator":
        from .evaluation.evaluator import Evaluator

        return Evaluator
    if name == "ModelCheckpoint":
        from .artifacts.checkpoint import ModelCheckpoint

        return ModelCheckpoint
    if name == "Trainer":
        from .training.trainer import Trainer

        return Trainer
    if name == "NNLogger":
        from .training.logger import NNLogger

        return NNLogger
    raise AttributeError(name)