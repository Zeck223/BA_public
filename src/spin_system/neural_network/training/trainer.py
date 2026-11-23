"""Train neural-network models with checkpointing and early stopping."""

import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..models import SpinResNet18


class Trainer:
    """Train a spin-grid model with mini-batches and early stopping.

    Early stopping triggers only after all output variables stop improving.

    Args:
        device: PyTorch device for model and batches.
        L: Height and width of each square input grid.
        num_out_features: Number of predicted values per sample.
        target_names: Optional names for model outputs.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per mini-batch.
        patience: Epochs without improvement before early stopping.
        lr: Initial learning rate.
        weight_decay: AdamW weight-decay coefficient.
        min_lr: Minimum learning rate for scheduling.
        warmup_epochs: Number of linear warmup epochs.
        dropout: Dropout probability before the output layer.
        print_every: Epoch interval for progress output.
        in_channels: Number of input channels per grid position.

    Returns:
        Trainer: A configured model trainer.
    """

    def __init__(
        self,
        device: torch.device,
        L: int,
        num_out_features: int,
        target_names: list = None,
        epochs: int = 1000,
        batch_size: int = 64,
        patience: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_epochs: int = 3,
        dropout: float = 0.0,
        print_every: int = 10,
        in_channels: int = 3
    ):
        """Initialize the model, optimizer, loss, and learning-rate schedule.

        Args:
            device: PyTorch device for model and batches.
            L: Height and width of each square input grid.
            num_out_features: Number of predicted values per sample.
            target_names: Optional names for output variables.
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per mini-batch.
            patience: Epochs without improvement before early stopping.
            lr: Initial learning rate.
            weight_decay: AdamW weight-decay coefficient.
            min_lr: Minimum learning rate for scheduling.
            warmup_epochs: Number of linear warmup epochs; zero disables warmup.
            dropout: Dropout probability before the output layer.
            print_every: Epoch interval for progress output.
            in_channels: Number of input channels per grid position.

        Returns:
            None: Initializes the training state.
        """
        self.device = device
        self.L = L
        self.num_out_features = num_out_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.dropout = dropout
        self.print_every = print_every
        self.in_channels = in_channels

        # Target names
        if target_names is None:
            self.target_names = [f"Output_{i}" for i in range(num_out_features)]
        else:
            self.target_names = target_names

        # Model initialization
        self.model = SpinResNet18(
            num_out_features,
            L=self.L,
            dropout=self.dropout,
            in_channels=in_channels,
        ).to(device)

        # AMP scaler (only active on CUDA)
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

        # Loss and Optimizer
        self.criterion = nn.HuberLoss(reduction='none')
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning Rate Scheduler: Linear Warmup + Cosine Annealing
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=min_lr / lr,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(epochs, 1),  # covers full training duration after warmup
                eta_min=min_lr
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=min_lr
            )

        # Training state
        self.best_val_losses = [float('inf')] * num_out_features
        self.best_avg_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
        self.epochs_without_improvement = [0] * num_out_features

        # Loss history per variable
        self.train_losses_per_var = [[] for _ in range(num_out_features)]
        self.val_losses_per_var = [[] for _ in range(num_out_features)]

    def _create_data_loader(self, dataset: Dataset, shuffle: bool = True):
        """Create a DataLoader for a simulation dataset.

        Args:
            dataset: Dataset to load in mini-batches.
            shuffle: Whether to shuffle sample order.

        Returns:
            DataLoader: Configured PyTorch data loader.
        """
        num_workers = min(4, os.cpu_count() or 0)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=num_workers > 0
        )

    def _train_epoch(
        self,
        train_loader,
        train_dataset_size,
        record=True,
    ):
        """Run one training epoch and optionally record per-target losses.

        Args:
            train_loader: Mini-batch loader for the training dataset.
            train_dataset_size: Number of training samples.
            record: Whether to append losses to the training history.

        Returns:
            None: Updates model parameters and optional loss history.
        """
        self.model.train()
        epoch_losses = torch.zeros(self.num_out_features, device=self.device)
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                out = self.model(X_batch)
                loss_per_sample = self.criterion(out, y_batch)
                total_loss = loss_per_sample.mean(dim=0).mean()

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss per variable (vectorized, stays on GPU)
            epoch_losses += loss_per_sample.sum(dim=0)

        # Average losses per variable
        if record:
            avg_losses = (epoch_losses / train_dataset_size).tolist()
            for i in range(self.num_out_features):
                self.train_losses_per_var[i].append(avg_losses[i])

    def _validate(
        self,
        val_loader,
        val_dataset_size,
        record=True,
    ):
        """Evaluate the model over validation batches.

        Args:
            val_loader: Mini-batch loader for validation data.
            val_dataset_size: Number of validation samples.
            record: Whether to append losses to the validation history.

        Returns:
            list[float]: Average validation loss for each output variable.
        """
        self.model.eval()
        epoch_losses = torch.zeros(self.num_out_features, device=self.device)
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    val_out = self.model(X_batch)
                    loss_per_sample = self.criterion(val_out, y_batch)

                # Accumulate loss per variable (vectorized, stays on GPU)
                epoch_losses += loss_per_sample.sum(dim=0)

        # Average losses per variable (single CPU transfer at end)
        val_losses_list = (epoch_losses / val_dataset_size).tolist()
        if record:
            for i in range(self.num_out_features):
                self.val_losses_per_var[i].append(val_losses_list[i])

        return val_losses_list

    def _check_early_stopping(self, val_losses_list, epoch):
        """Update best-validation state and evaluate the stopping condition.

        Args:
            val_losses_list: Current validation losses, one per output variable.
            epoch: Current zero-based epoch index.

        Returns:
            bool: Whether all outputs have reached the patience limit.
        """
        avg_val_loss = sum(val_losses_list) / len(val_losses_list)

        for i in range(self.num_out_features):
            if val_losses_list[i] < self.best_val_losses[i]:
                self.best_val_losses[i] = val_losses_list[i]
                self.epochs_without_improvement[i] = 0
            else:
                self.epochs_without_improvement[i] += 1

        # Save model only if the average loss has improved
        if avg_val_loss < self.best_avg_val_loss:
            self.best_avg_val_loss = avg_val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.best_epoch = epoch
            best_str = ", ".join(
                f"{self.target_names[i]}={self.best_val_losses[i]:.6f}"
                for i in range(self.num_out_features)
            )

        # Early stopping only when ALL variables show no improvement
        return all(e >= self.patience for e in self.epochs_without_improvement)

    def _print_progress(self, epoch, losses):
        """Print the current validation losses and learning rate.

        Args:
            epoch: Current epoch index.
            losses: Validation loss for each output variable.

        Returns:
            None: Writes progress to the console.
        """
        current_lr = self.optimizer.param_groups[0]['lr']
        loss_str = ", ".join([
            f"{self.target_names[i]}={losses[i]:.6f}"
            for i in range(self.num_out_features)
        ])
        print(f"Epoch {epoch}: val_losses: [{loss_str}], lr = {current_lr:.8f}")


    def _load_best_model(self):
        """Restore the model state with the best average validation loss.

        Args:
            None: Uses the best state stored during training.

        Returns:
            None: Restores the model in place when a best state exists.
        """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            best_str = ", ".join(
                f"{self.target_names[i]}={self.best_val_losses[i]:.6f}"
                for i in range(self.num_out_features)
            )
            print(f"\nBest model loaded from epoch {self.best_epoch - self.warmup_epochs} [{best_str}]")


    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> tuple:
        """Train, validate, schedule learning rates, and restore the best model.

        Args:
            train_dataset: Training samples and targets.
            val_dataset: Validation samples and targets.

        Returns:
            tuple: Model, optimizer, per-target training losses, per-target
            validation losses, and target names.
        """
        train_loader = self._create_data_loader(train_dataset)
        val_loader = self._create_data_loader(val_dataset, shuffle=False)
        train_dataset_size = len(train_dataset)
        val_dataset_size = len(val_dataset)

        print("\n" + "-" * 60)
        print(f"Training:     {SpinResNet18.__name__}")
        print(f"Device:       {self.device}")
        print(f"Epochs:       {self.epochs}")
        print(f"Patience:     {self.patience}")
        print(f"Batch size:   {self.batch_size}")
        print(f"LR:           {self.lr}")
        print(f"Warmup:       {self.warmup_epochs} epochs")
        print(f"Targets:      {', '.join(self.target_names)}")
        print("-" * 60 + "\n")

        should_stop = False
        for epoch in range(self.epochs + self.warmup_epochs):
            in_warmup = epoch < self.warmup_epochs

            # === Training Phase ===
            self._train_epoch(
                train_loader,
                train_dataset_size,
                record=not in_warmup,
            )

            # === Validation Phase ===
            val_losses_list = self._validate(
                val_loader,
                val_dataset_size,
                record=not in_warmup,
            )
            self.scheduler.step()

            if in_warmup:
                if (epoch + 1) % self.print_every == 0:
                    print(f"[Warmup {epoch + 1:3d}/{self.warmup_epochs}]", end="\r")
            else:
                if epoch > self.epochs // 5:
                    should_stop = self._check_early_stopping(val_losses_list, epoch)

                if epoch % self.print_every == 0:
                    self._print_progress(epoch, val_losses_list)

                if should_stop:
                    print(f"\nEarly stopping after {epoch + 1 - self.warmup_epochs} epochs")
                    print(
                        "Epochs without improvement: "
                        f"{dict(zip(self.target_names, self.epochs_without_improvement))}"
                    )
                    break

        self._load_best_model()

        return (
            self.model,
            self.optimizer,
            self.train_losses_per_var,
            self.val_losses_per_var,
            self.target_names
        )
