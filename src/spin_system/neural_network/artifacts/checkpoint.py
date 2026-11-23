"""Save and restore PyTorch model checkpoints."""

import torch
from pathlib import Path
from typing import Optional


class ModelCheckpoint:
    """Persist model, optimizer, and experiment metadata together.

    Args:
        model: Optional PyTorch model whose state should be stored.
        optimizer: Optional optimizer whose state should be stored.

    Returns:
        ModelCheckpoint: A checkpoint manager instance.
    """

    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None
        ) -> None:
        """Bind optional model and optimizer objects to the checkpoint manager.

        Args:
            model: Model whose state should be saved or restored.
            optimizer: Optimizer whose state should be saved or restored.

        Returns:
            None: Stores the objects for later checkpoint operations.
        """
        self.model = model
        self.optimizer = optimizer

    def save(
        self,
        path: str,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        **extra_data
    ) -> None:
        """Save available states and additional metadata to a checkpoint file.

        Args:
            path: Destination path for the checkpoint.
            epoch: Optional training epoch to record.
            loss: Optional loss value to record.
            extra_data: Additional checkpoint metadata.

        Returns:
            None: Writes the checkpoint to disk.
        """
        checkpoint = {}
        if self.model is not None:
            checkpoint["model_state_dict"] = self.model.state_dict()

        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if loss is not None:
            checkpoint["loss"] = loss

        checkpoint.update(extra_data)

        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, path)

    def load(
        self,
        path: str,
        map_location: Optional[str] = "cpu"
    ) -> dict:
        """Load a checkpoint and restore its model and optimizer states.

        Args:
            path: Checkpoint file to load.
            map_location: Device on which checkpoint tensors are loaded.

        Returns:
            dict: Loaded checkpoint data.
        """
        checkpoint = torch.load(path, map_location=map_location)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
