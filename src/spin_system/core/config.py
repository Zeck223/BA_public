"""Configuration loading shared by generation and model workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import ProjectPaths


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from a file.

    Args:
        path: YAML file path.

    Returns:
        dict[str, Any]: Loaded mapping, or an empty mapping for an empty file.
    """
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config or {}


class NNConfigLoader:
    """Load default NN settings and apply a dataset-specific override.

    Args:
        paths: Optional project paths used to locate the NN settings file.

    Returns:
        NNConfigLoader: A configuration loader instance.
    """

    def __init__(self, paths: ProjectPaths | None = None) -> None:
        """Initialize the loader with explicit or default project paths.

        Args:
            paths: Optional project paths.

        Returns:
            None: Stores the paths used for subsequent loads.
        """
        self.paths = paths or ProjectPaths()

    def load(self, config_name: str) -> dict[str, Any]:
        """Merge default NN settings with one named configuration override.

        Args:
            config_name: Configuration key whose overrides should be applied.

        Returns:
            dict[str, Any]: Deep-copied merged NN settings.
        """
        config = load_yaml(self.paths.nn_config)
        values = deepcopy(config.get("default", {}))
        values.update(deepcopy(config.get("configs", {}).get(config_name, {})))
        return values