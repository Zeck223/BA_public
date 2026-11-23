"""Lazy datasets for generated simulation arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ...core.file_manager import FileManager
from .normalization import normalize_input_array


class SimDataset(Dataset):
    """Load selected data sources only when a sample is requested.

    Args:
        file_manager: Manager used to load simulation arrays.
        config_name: Configuration containing the requested simulations.
        sim_names: Ordered simulation names represented by this dataset.
        targets: Target values corresponding to ``sim_names``.
        data_sources: Data sources to concatenate for each sample.

    Returns:
        SimDataset: A lazy dataset over simulation arrays.
    """

    def __init__(
        self,
        file_manager: FileManager,
        config_name: str,
        sim_names: Sequence[str],
        targets: np.ndarray,
        data_sources: Sequence[str] | None = None,
    ) -> None:
        """Store sample metadata and target values.

        Args:
            file_manager: Manager used to load simulation arrays.
            config_name: Configuration containing the requested simulations.
            sim_names: Ordered simulation names represented by this dataset.
            targets: Target values corresponding to ``sim_names``.
            data_sources: Data sources to concatenate for each sample.

        Returns:
            None: Initializes the dataset metadata.
        """
        self.file_manager = file_manager
        self.config_name = config_name
        self.sim_names = list(sim_names)
        self.y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.data_sources = tuple(data_sources or ("grids",))

    def __len__(self) -> int:
        """Return the number of simulations in the dataset.

        Args:
            None: Uses the stored simulation-name list.

        Returns:
            int: Number of samples.
        """
        return len(self.sim_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize one sample and pair it with its target.

        Args:
            idx: Index of the requested simulation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Flattened input and target tensors.
        """
        sim_name = self.sim_names[idx]
        arrays = [
            normalize_input_array(
                self.file_manager.load_array(
                    self.config_name,
                    sim_name,
                    source=source,
                ),
                source,
            )
            for source in self.data_sources
        ]
        values = np.concatenate(arrays, axis=-1).flatten().astype(np.float32)
        return torch.from_numpy(values), self.y[idx]