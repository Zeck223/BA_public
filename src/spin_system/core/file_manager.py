"""Common file access for generated data and model workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .paths import ProjectPaths


class FileManager:
    """Read and write project data without relying on the working directory."""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        project_root: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize project paths for all file operations.

        Args:
            paths: Optional preconfigured project paths.
            project_root: Optional project root.
            data_root: Optional data root.

        Returns:
            None: Stores the resolved paths.

        Raise:
            ValueError: If paths is combined with project_root or data_root.
        """
        if paths is not None and (project_root is not None or data_root is not None):
            raise ValueError("Pass either paths or project/data roots, not both")
        self.paths = paths or ProjectPaths(project_root=project_root, data_root=data_root)

    def _config_candidates(self, config_name: str) -> Iterable[Path]:
        """Yield supported storage locations for a configuration.

        Args:
            config_name: Name of the configuration to locate.

        Returns:
            Iterable[Path]: Candidate paths in lookup order.
        """
        yield self.paths.data_root / config_name / f"{config_name}.npy"
        yield self.paths.data_root / config_name / f"{config_name}.yaml"
        yield self.paths.parameter_config_root / f"{config_name}.npy"
        yield self.paths.parameter_config_root / f"{config_name}.yaml"
        yield self.paths.config_root / f"{config_name}.npy"
        yield self.paths.config_root / f"{config_name}.yaml"

    def load_config(self, config_name: str) -> dict[str, Any]:
        """Load a config from data storage, then generated config storage.

        Args:
            config_name: Name of the configuration to load.

        Returns:
            dict[str, Any]: Loaded configuration mapping.

        Raise:
            ValueError: If a stored configuration is not a mapping.
            FileNotFoundError: If no candidate configuration file exists.
        """
        candidates = list(self._config_candidates(config_name))
        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix == ".npy":
                config = np.load(path, allow_pickle=True).item()
                if not isinstance(config, dict):
                    raise ValueError(f"Configuration at {path} is not a mapping")
                return config
            with path.open(encoding="utf-8") as file:
                config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError(f"Configuration at {path} is not a mapping")
            return config

        formatted_paths = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Configuration '{config_name}' not found. Checked: {formatted_paths}")

    def save_config(self, config: dict[str, Any], config_name: str | None = None) -> Path:
        """Persist a config in YAML and NumPy formats.

        Args:
            config: Configuration mapping to save.
            config_name: Optional name overriding the config's own name.

        Returns:
            Path: Path to the saved YAML configuration.
        """
        name = config_name or config["config_name"]
        config_directory = self.paths.config_directory(name)
        config_directory.mkdir(parents=True, exist_ok=True)

        yaml_path = config_directory / f"{name}.yaml"
        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        np.save(config_directory / f"{name}.npy", config)
        return yaml_path

    def save_parameter_config(self, config: dict[str, Any]) -> Path:
        """Persist generated parameters in the canonical configuration folder.

        Args:
            config: Generated parameter configuration to save.

        Returns:
            Path: Path to the saved YAML configuration.
        """
        name = config["config_name"]
        self.paths.parameter_config_root.mkdir(parents=True, exist_ok=True)
        yaml_path = self.paths.parameter_config_root / f"{name}.yaml"
        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        np.save(self.paths.parameter_config_root / f"{name}.npy", config)
        return yaml_path

    def array_path(self, config_name: str, sim_name: str, source: str = "grids") -> Path:
        """Build the path for one simulation array.

        Args:
            config_name: Name of the simulation configuration.
            sim_name: Name of the simulation.
            source: Data source directory.

        Returns:
            Path: Path to the simulation array file.
        """
        return self.paths.source_directory(config_name, source) / f"{sim_name}.npy"

    def load_array(
        self,
        config_name: str,
        sim_name: str,
        source: str = "grids",
        allow_pickle: bool = False,
    ) -> np.ndarray:
        """Load one simulation array from a named data source.

        Args:
            config_name: Name of the simulation configuration.
            sim_name: Name of the simulation.
            source: Data source directory.
            allow_pickle: Whether NumPy may load pickled objects.

        Returns:
            np.ndarray: Loaded simulation data.

        Raise:
            FileNotFoundError: If the simulation array does not exist.
        """
        path = self.array_path(config_name, sim_name, source)
        if path.is_file():
            return np.load(path, allow_pickle=allow_pickle)
        raise FileNotFoundError(f"Simulation data not found: {path}")

    def save_array(
        self,
        config_name: str,
        sim_name: str,
        array: np.ndarray,
        source: str = "grids",
    ) -> Path:
        """Save one simulation array and return its path.

        Args:
            config_name: Name of the simulation configuration.
            sim_name: Name of the simulation.
            array: Data to store.
            source: Data source directory.

        Returns:
            Path: Path to the saved array.
        """
        directory = self.paths.source_directory(config_name, source)
        directory.mkdir(parents=True, exist_ok=True)
        path = self.array_path(config_name, sim_name, source)
        np.save(path, array)
        return path

    def save_spins(self, config_name: str, sim_name: str, spins: np.ndarray) -> Path:
        """Save one spin grid in the default data source.

        Args:
            config_name: Name of the simulation configuration.
            sim_name: Name of the simulation.
            spins: Spin grid to store.

        Returns:
            Path: Path to the saved spin grid.
        """
        return self.save_array(config_name, sim_name, spins, source="grids")

    def list_simulations(self, config_name: str, source: str = "grids") -> list[str]:
        """Return sorted simulation names for a data source.

        Args:
            config_name: Name of the simulation configuration.
            source: Data source directory.

        Returns:
            list[str]: Sorted simulation names, or an empty list if absent.
        """
        directory = self.paths.source_directory(config_name, source)
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.npy"))