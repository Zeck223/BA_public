"""Dataset and batch construction for generated simulation data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ...core.file_manager import FileManager
from .dataset import SimDataset
from .schema import get_simulation_parameter_names, get_target_columns


class SimulationDataService:
    """Provide one data-access boundary for generation and NN workflows.

    Args:
        config_name: Configuration whose simulation data should be accessed.
        file_manager: Optional shared manager for data and paths.
        data_sources: Data sources to load and concatenate.

    Returns:
        SimulationDataService: A data-access service instance.
    """

    def __init__(
        self,
        config_name: str,
        file_manager: FileManager | None = None,
        data_sources: Sequence[str] | None = None,
    ) -> None:
        """Load configuration metadata and select data sources.

        Args:
            config_name: Configuration whose simulation data should be accessed.
            file_manager: Optional shared manager for data and paths.
            data_sources: Data sources to load and concatenate.

        Returns:
            None: Initializes the service state.
        """
        self.config_name = config_name
        self.file_manager = file_manager or FileManager()
        self.data_sources = tuple(data_sources or ("grids",))
        self.config = self.file_manager.load_config(config_name)

    def get_config(self) -> dict:
        """Return the configuration loaded by this service.

        Args:
            None: Uses the configuration stored on this instance.

        Returns:
            dict: Loaded simulation configuration.
        """
        return self.config

    def get_sim_param_names(self) -> list[str]:
        """Return the parameter names available in the configuration.

        Args:
            None: Uses the configuration stored on this instance.

        Returns:
            list[str]: Simulation parameter names.
        """
        return get_simulation_parameter_names(self.config)

    def build_lazy_dataset(
        self,
        sim_names: Sequence[str],
        targets: np.ndarray,
    ) -> SimDataset:
        """Build a lazy dataset from ordered simulation names and targets.

        Args:
            sim_names: Ordered simulation names represented by the dataset.
            targets: Target values corresponding to ``sim_names``.

        Returns:
            SimDataset: Dataset that loads simulation arrays on demand.
        """
        return SimDataset(
            self.file_manager,
            self.config_name,
            sim_names,
            targets,
            data_sources=self.data_sources,
        )

    def build_lazy_datasets(
        self,
        train_size: float,
        seed: int,
        save_path: str | Path,
    ) -> tuple[list[str], SimDataset, SimDataset]:
        """Build train and validation datasets without loading grids into RAM.

        Args:
            train_size: Fraction of simulations assigned to the training set.
            seed: Random seed used for the train/validation split.
            save_path: Directory where the fitted target scaler is stored.

        Returns:
            tuple[list[str], SimDataset, SimDataset]: Target names, training
            dataset, and validation dataset.
        """
        sim_names = sorted(self.config.get("parameter_sets", {}).keys())
        parameter_sets = self.config["parameter_sets"]
        parameter_names = self.get_sim_param_names()
        target_columns = get_target_columns(self.config, parameter_names)

        print(f"Target columns: {target_columns}")
        targets = np.array(
            [[parameter_sets[name][column] for column in target_columns] for name in sim_names],
            dtype=np.float32,
        )

        indices = list(range(len(sim_names)))
        train_indices, validation_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
        )
        train_names = [sim_names[index] for index in train_indices]
        validation_names = [sim_names[index] for index in validation_indices]

        scaler_y = MinMaxScaler()
        train_targets = scaler_y.fit_transform(targets[train_indices])
        validation_targets = scaler_y.transform(targets[validation_indices])

        save_directory = Path(save_path)
        save_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_y, save_directory / "scaler_y.pkl")

        print(f"Dataset: {len(train_names)} train / {len(validation_names)} val samples")
        return (
            target_columns,
            self.build_lazy_dataset(train_names, train_targets),
            self.build_lazy_dataset(validation_names, validation_targets),
        )