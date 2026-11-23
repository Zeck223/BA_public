"""Shared parameter and target-column rules for training and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def display_parameter_name(name: str) -> str:
    """Return the mathematical label used for a parameter in figures.

    Args:
        name: Parameter name to display.

    Returns:
        str: Display label.
    """
    return r"$\beta$" if name == "DMI" else name


def get_simulation_parameter_names(config: dict[str, Any]) -> list[str]:
    """Return parameter names present in the generated simulation config.

    Args:
        config: Generated simulation configuration.

    Returns:
        list[str]: Sorted parameter names from the first parameter set.
    """
    parameter_sets = config.get("parameter_sets", {})
    if not parameter_sets:
        return []
    first_parameters = next(iter(parameter_sets.values()))
    return sorted(first_parameters)


def get_target_columns(
    config: dict[str, Any],
    parameter_names: Iterable[str] | None = None,
) -> list[str]:
    """Return model targets in the established project order.

    ``J`` remains an input parameter rather than a prediction target.

    Args:
        config: Simulation configuration with parameter ranges and sets.
        parameter_names: Optional set of available parameter names.

    Returns:
        list[str]: Target parameter names, excluding ``J``.
    """
    parameter_names = set(parameter_names or get_simulation_parameter_names(config))
    possible_targets = list(config.get("parameter_ranges", {}).keys())
    return [name for name in possible_targets if name in parameter_names and name != "J"]


def resolve_target_columns(
    target_columns: Iterable[str],
    available_columns: Iterable[str],
) -> list[str]:
    """Validate checkpoint target names and retain their output order.

    Args:
        target_columns: Ordered target names stored in the checkpoint.
        available_columns: Dataset column names available for evaluation.

    Returns:
        list[str]: Available column names matching the checkpoint target order.

    Raise:
        ValueError: If a checkpoint target is missing from the dataset.
    """
    available = set(available_columns)
    resolved_columns = []
    for target in target_columns:
        if target not in available:
            raise ValueError(
                f"Target column {target!r} from the model checkpoint is missing "
                f"from the evaluation dataset."
            )
        resolved_columns.append(target)
    return resolved_columns
