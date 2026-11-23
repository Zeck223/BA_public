"""Project, resource, data, and artifact path resolution."""

from __future__ import annotations

import os
from pathlib import Path


class ProjectPaths:
    """Resolve project paths without making the working directory implicit.

    Args:
        project_root: Optional project-root override.
        data_root: Optional data-root override.

    Returns:
        ProjectPaths: A resolver for configuration, output, and data paths.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        """Resolve the project and data roots from arguments or environment.

        Args:
            project_root: Optional project-root override.
            data_root: Optional data-root override.

        Returns:
            None: Stores the resolved roots on this instance.
        """
        self.project_root = self._resolve_project_root(project_root)
        if project_root is None and data_root is None:
            self._load_dotenv()

        configured_data_root = data_root
        if configured_data_root is None:
            configured_data_root = os.environ.get("SPIN_SYSTEM_DATA_ROOT")
        if configured_data_root:
            self.data_root = self._resolve_configured_path(configured_data_root)
        else:
            self.data_root = self.project_root / "data"

    def _load_dotenv(self) -> None:
        """Load project environment defaults without overriding shell values.

        Args:
            None: Reads the project-level ``.env`` file.

        Returns:
            None: Adds missing environment variables.
        """
        env_path = self.project_root / ".env"
        if not env_path.is_file():
            return

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip("\"'")
            if name:
                os.environ.setdefault(name, value)

    def _resolve_configured_path(self, value: str | Path) -> Path:
        """Resolve a configured path relative to the project root if needed.

        Args:
            value: Configured path as text or a ``Path``.

        Returns:
            Path: Expanded and absolute path.
        """
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @staticmethod
    def _resolve_project_root(project_root: str | Path | None) -> Path:
        """Choose the project root from an override or available defaults.

        Args:
            project_root: Optional explicit project-root path.

        Returns:
            Path: Resolved project root.
        """
        if project_root is not None:
            return Path(project_root).expanduser().resolve()

        configured_root = os.environ.get("SPIN_SYSTEM_ROOT")
        if configured_root:
            return Path(configured_root).expanduser().resolve()

        current_directory = Path.cwd().resolve()
        if (current_directory / "pyproject.toml").is_file():
            return current_directory

        source_root = Path(__file__).resolve().parents[3]
        if (source_root / "pyproject.toml").is_file():
            return source_root
        return current_directory

    @property
    def config_root(self) -> Path:
        """Return the directory containing canonical configuration files.

        Args:
            None: Uses the configured environment or project root.

        Returns:
            Path: Canonical configuration directory.
        """
        configured_root = os.environ.get("SPIN_SYSTEM_CONFIG_ROOT")
        return (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else self.project_root / "configs"
        )

    @property
    def package_config_root(self) -> Path:
        """Return the configuration-resource directory bundled with the package.

        Args:
            None: Resolves the installed package location.

        Returns:
            Path: Bundled configuration directory.
        """
        return Path(__file__).resolve().parents[1] / "configs"

    @property
    def parameter_config_root(self) -> Path:
        """Return the directory containing generated parameter configurations.

        Args:
            None: Uses the canonical configuration directory.

        Returns:
            Path: Generated parameter configuration directory.
        """
        return self.config_root / "parameter_configs"

    @staticmethod
    def _first_existing(*candidates: Path) -> Path:
        """Return the first existing file, or the first candidate as fallback.

        Args:
            candidates: Ordered candidate paths to inspect.

        Returns:
            Path: First existing candidate, or the first supplied candidate.
        """
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @property
    def simulation_config(self) -> Path:
        """Return the parameter-generator configuration path.

        Args:
            None: Searches canonical and bundled configuration locations.

        Returns:
            Path: Selected simulation configuration path.
        """
        return self._first_existing(
            self.config_root / "simulation.yaml",
            self.config_root / "simulation_config.yaml",
            self.package_config_root / "simulation.yaml",
        )

    @property
    def nn_config(self) -> Path:
        """Return the neural-network configuration path.

        Args:
            None: Searches canonical and bundled configuration locations.

        Returns:
            Path: Selected neural-network configuration path.
        """
        return self._first_existing(
            self.config_root / "neural_network.yaml",
            self.config_root / "nn_config.yaml",
            self.package_config_root / "neural_network.yaml",
        )

    @property
    def models_root(self) -> Path:
        """Return the root directory for model checkpoints.

        Args:
            None: Uses the configured environment or project root.

        Returns:
            Path: Model storage directory.
        """
        configured_root = os.environ.get("SPIN_SYSTEM_MODEL_ROOT")
        return self._resolve_configured_path(configured_root or "models")

    @property
    def evaluation_root(self) -> Path:
        """Return the root directory for evaluation outputs.

        Args:
            None: Uses the configured output root.

        Returns:
            Path: Evaluation output directory.
        """
        return self.output_root / "evaluation"

    @property
    def plots_root(self) -> Path:
        """Return the root directory for individual plots.

        Args:
            None: Uses the configured output root.

        Returns:
            Path: Plot output directory.
        """
        return self.output_root / "plots"

    @property
    def mosaic_root(self) -> Path:
        """Return the root directory for mosaic plots.

        Args:
            None: Uses the configured output root.

        Returns:
            Path: Mosaic output directory.
        """
        return self.output_root / "mosaic"

    @property
    def videos_root(self) -> Path:
        """Return the root directory for generated videos.

        Args:
            None: Uses the configured output root.

        Returns:
            Path: Video output directory.
        """
        return self.output_root / "videos"

    @property
    def output_root(self) -> Path:
        """Return the configured output root.

        Args:
            None: Uses the configured environment or project root.

        Returns:
            Path: Root directory for generated output.
        """
        configured_root = os.environ.get("SPIN_SYSTEM_OUTPUT_ROOT")
        return self._resolve_configured_path(configured_root or "outputs")

    def config_directory(self, config_name: str) -> Path:
        """Return the data directory for a configuration.

        Args:
            config_name: Name of the configuration.

        Returns:
            Path: Configuration data directory.
        """
        return self.data_root / config_name

    def source_directory(self, config_name: str, source: str = "grids") -> Path:
        """Return a data-source directory for a configuration.

        Args:
            config_name: Name of the configuration.
            source: Name of the data source.

        Returns:
            Path: Selected source directory.
        """
        return self.config_directory(config_name) / source

    def model_directory(self, config_name: str, model_name: str) -> Path:
        """Return the directory for one named model.

        Args:
            config_name: Name of the simulation configuration.
            model_name: Name of the model.

        Returns:
            Path: Model directory.
        """
        return self.models_root / config_name / model_name

    def evaluation_directory(
        self,
        config_name: str,
        model_name: str,
        test_set_name: str,
    ) -> Path:
        """Return the output directory for one model evaluation.

        Args:
            config_name: Name of the simulation configuration.
            model_name: Name of the model.
            test_set_name: Name of the evaluated test set.

        Returns:
            Path: Evaluation directory.
        """
        return self.evaluation_root / config_name / model_name / test_set_name



