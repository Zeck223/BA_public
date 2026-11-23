"""Public, lazily imported APIs for simulation generation and visualization."""

from typing import Any


__all__ = [
    "Simulation",
    "create_config",
    "generate_grid_parameters",
    "generate_random_parameters",
    "load_simulation_config",
    "validate_simulation_config",
]


def __getattr__(name: str) -> Any:
    """Resolve public generation APIs only when they are first accessed.

    Args:
        name: Public attribute name requested from this package.

    Returns:
        Any: The matching class or function object.

    Raise:
        AttributeError: If the requested name is not exported by this package.
    """

    if name == "Simulation":
        from .simulation import Simulation

        return Simulation
    if name in {
        "create_config",
        "generate_grid_parameters",
        "generate_random_parameters",
        "load_simulation_config",
        "validate_simulation_config",
    }:
        from .generator import (
            create_config,
            generate_grid_parameters,
            generate_random_parameters,
            load_simulation_config,
            validate_simulation_config,
        )

        return {
            "create_config": create_config,
            "generate_grid_parameters": generate_grid_parameters,
            "generate_random_parameters": generate_random_parameters,
            "load_simulation_config": load_simulation_config,
            "validate_simulation_config": validate_simulation_config,
        }[name]
    raise AttributeError(name)
