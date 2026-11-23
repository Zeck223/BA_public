"""Evaluate trained models and create prediction and loss visualizations."""

import numpy as np
import pandas as pd
import torch
import joblib
import seaborn as sns
from itertools import combinations
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from spin_system import FileManager
from ..artifacts.checkpoint import ModelCheckpoint
from ..data import SimDataset, SimulationDataService
from ..data.schema import (
    display_parameter_name,
    get_target_columns,
    resolve_target_columns,
)
from ..models import SpinResNet18


class Evaluator:
    """Load model predictions and generate evaluation summaries and plots.

    Args:
        config_name: Configuration used to locate model and evaluation outputs.
        model: Optional preloaded model.
        y_pred: Optional prediction matrix.
        y_true: Optional ground-truth matrix.
        target_cols: Optional ordered target-column names.
        model_save_name: Model directory name.
        test_set_name: Test-set configuration name.
        data_sources: Input sources used by the model.
        file_manager: Optional manager for configuration, data, and paths.

    Returns:
        Evaluator: An evaluator instance.
    """

    def __init__(
            self,
            config_name,
            model=None,
            y_pred=None,
            y_true=None,
            target_cols=None,
            model_save_name=None,
            test_set_name=None,
            data_sources: list = None,
            file_manager: FileManager | None = None,
        ) -> None:
        """Initialize evaluation state and load predictions when needed.

        Args:
            config_name: Configuration used to locate model and outputs.
            model: Optional preloaded model.
            y_pred: Optional prediction matrix.
            y_true: Optional ground-truth matrix.
            target_cols: Optional ordered target names.
            model_save_name: Model directory name.
            test_set_name: Test-set configuration name.
            data_sources: Input sources used by the model.
            file_manager: Optional manager for configuration, data, and paths.

        Returns:
            None: Initializes the model and evaluation data.
        """
        self.config_name = config_name
        self.model_save_name = model_save_name
        self.test_set_name = test_set_name
        self.data_sources = data_sources if data_sources is not None else ["grids"]
        self.file_manager = file_manager or FileManager()
        self.config = self.file_manager.load_config(config_name)
        self.target_cols = list(self.config['parameter_ranges'].keys())
        self.model = self._init_model() if model is None else model
        if target_cols is not None:
            self.target_cols = target_cols
        self.save_path = self.file_manager.paths.evaluation_directory(
            self.config_name,
            self.model_save_name,
            self.test_set_name,
        )
        self.save_path.mkdir(parents=True, exist_ok=True)

        if y_pred is not None and y_true is not None:
            self.y_pred = y_pred
            self.y_true = y_true
            self._order_targets_for_plots()
        else:
            self._init_data()

    @property
    def y_pred(self):
        """Return predictions, initializing evaluation data on first access.

        Args:
            None: Uses this evaluator's configured data.

        Returns:
            np.ndarray: Prediction matrix in original target units.
        """
        if not hasattr(self, "_y_pred"):
            self._init_data()
        return self._y_pred

    @y_pred.setter
    def y_pred(self, value):
        """Set the prediction matrix used by evaluation methods.

        Args:
            value: Prediction matrix in original target units.

        Returns:
            None: Stores the prediction matrix.
        """
        self._y_pred = value

    def _init_model(self) -> SpinResNet18:
        """Load the trained model and checkpoint metadata from disk.

        Args:
            None: Uses model and configuration names stored on this instance.

        Returns:
            SpinResNet18: Restored model architecture and checkpoint weights.

        Raise:
            ValueError: If checkpoint output metadata is incompatible with config.
        """
        print("Loading model...")
        model_directory = self.file_manager.paths.model_directory(
            self.config_name,
            self.model_save_name,
        )
        model_path = model_directory / "checkpoint.pth"

        # Peek at checkpoint to determine actual output size, target names and data sources
        raw = torch.load(model_path, map_location="cpu")
        state = raw["model_state_dict"]
        n_outputs = state["fc.weight"].shape[0]
        if "target_cols" in raw:
            self.target_cols = raw["target_cols"]
        else:
            self.target_cols = get_target_columns(self.config)
            if len(self.target_cols) != n_outputs:
                raise ValueError(
                    "Checkpoint has no target_cols metadata, and its output "
                    f"count ({n_outputs}) does not match the configured target "
                    f"count ({len(self.target_cols)})."
                )

        # Determine in_channels from stored data_sources (fallback: ["grids"])
        checkpoint_sources = raw.get("data_sources", ["grids"])
        in_channels = len(checkpoint_sources) * 3
        # Keep evaluator's data_sources in sync with what the model was trained on
        self.data_sources = checkpoint_sources

        base_model = SpinResNet18(
            n_outputs,
            self.config["base_parameters"]["L"],
            dropout=0.0,
            in_channels=in_channels,
        )
        ModelCheckpoint(base_model).load(model_path)
        return base_model


    def _init_data(self) -> None:
        """Load evaluation data, predict targets, and restore their scale.

        Args:
            None: Uses the configured test set and model.

        Returns:
            None: Stores aligned predictions and targets on this instance.
        """
        print("Initializing data...")
        data_service = SimulationDataService(
            self.test_set_name,
            file_manager=self.file_manager,
            data_sources=self.data_sources,
        )
        parameter_sets = data_service.get_config()["parameter_sets"]
        self.target_cols = resolve_target_columns(
            self.target_cols,
            data_service.get_sim_param_names(),
        )
        sim_names = sorted(parameter_sets)
        targets = np.array(
            [
                [parameter_sets[name][target] for target in self.target_cols]
                for name in sim_names
            ],
            dtype=np.float64,
        )
        dataset = data_service.build_lazy_dataset(sim_names, targets)
        self.y_true = targets

        scaler_y = self._load_target_scaler()

        y_pred = self._predict(dataset)
        self.y_pred = scaler_y.inverse_transform(y_pred)
        self._order_targets_for_plots()

    def _order_targets_for_plots(self) -> None:
        """Order target data consistently for plots without changing model output.

        Args:
            None: Uses the current target names, predictions, and targets.

        Returns:
            None: Reorders plot data in place.
        """
        target_priority = {"DMI": 0, "D": 1, "K": 2}
        original_target_cols = list(self.target_cols)
        self._target_order_indices = sorted(
            range(len(original_target_cols)),
            key=lambda index: (
                target_priority.get(original_target_cols[index], len(target_priority)),
                index,
            ),
        )
        self.target_cols = [
            original_target_cols[index]
            for index in self._target_order_indices
        ]
        self.y_true = self.y_true[:, self._target_order_indices]
        self.y_pred = self.y_pred[:, self._target_order_indices]

    def _load_ordered_losses(self, loss_load_path: Path, split: str) -> np.ndarray:
        """Load per-target losses in the order used by target-specific plots.

        Args:
            loss_load_path: Directory containing saved loss arrays.
            split: Dataset split name, such as ``Train`` or ``Val``.

        Returns:
            np.ndarray: Per-target losses in plot order.
        """
        losses = np.load(
            loss_load_path / f"{split.lower()}_losses_per_var.npy",
            allow_pickle=True,
        )
        return losses[self._target_order_indices]


    def _predict(self, dataset: SimDataset, batch_size: int = 256) -> np.ndarray:
        """Make lazy dataset predictions in mini-batches.

        Args:
            dataset: Simulation dataset loaded on demand.
            batch_size: Number of input rows processed per model call.

        Returns:
            np.ndarray: Predicted target values in scaled units.
        """
        self.model.to(device="cpu")
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, _ in DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            ):
                outputs.append(self.model(inputs).numpy())
        return np.concatenate(outputs, axis=0)


    def _load_target_scaler(self):
        """Load the target scaler saved during training.

        Args:
            None: Uses this evaluator's model directory.

        Returns:
            MinMaxScaler: Scaler used to transform target values.

        Raise:
            FileNotFoundError: If the saved target scaler is missing.
        """
        scaler_dir = self.file_manager.paths.model_directory(
            self.config_name,
            self.model_save_name,
        ) / "scalers"
        scaler_y_path = scaler_dir / "scaler_y.pkl"

        if not scaler_y_path.exists():
            raise FileNotFoundError(
                f"Target scaler not found. Expected {scaler_y_path}. "
                "Run training first to generate scalers."
            )

        return joblib.load(scaler_y_path)


    def print_evaluation_metrics(self) -> None:
        """Print MSE, MAE, and R-squared metrics for each output variable.

        Args:
            None: Uses the prediction and target arrays stored on this instance.

        Returns:
            None: Prints per-target and aggregate metrics.
        """

        print(f"\n=== Test-Set Evaluation ===")
        print(f"{'Variable':<15} {'MSE':>10} {'MAE':>10} {'R²':>10}")
        print("-" * 47)
        for i, name in enumerate(self.target_cols):
            mse = mean_squared_error(self.y_true[:, i], self.y_pred[:, i])
            mae = mean_absolute_error(self.y_true[:, i], self.y_pred[:, i])
            r2 = r2_score(self.y_true[:, i], self.y_pred[:, i])
            print(f"{name:<15} {mse:>10.6f} {mae:>10.6f} {r2:>10.4f}")

        print("-" * 47)
        mse_total = mean_squared_error(self.y_true, self.y_pred)
        mae_total = mean_absolute_error(self.y_true, self.y_pred)
        r2_total = r2_score(self.y_true, self.y_pred)
        print(f"{'TOTAL':<15} {mse_total:>10.6f} {mae_total:>10.6f} {r2_total:>10.4f}")


    def eval_pred_vs_true(self) -> None:
        """Plot true versus predicted values for each target.

        Args:
            None: Uses the prediction and target arrays stored on this instance.

        Returns:
            None: Saves one comparison plot per target.
        """

        for idx, col_name in enumerate(self.target_cols):

            y_true_feat = self.y_true[:, idx]
            y_pred_feat = self.y_pred[:, idx]

            r2 = r2_score(y_true_feat, y_pred_feat)

            plt.figure(figsize=(10, 10))
            plt.scatter(y_true_feat, y_pred_feat, alpha=0.5, marker="x")
            plt.plot([y_true_feat.min(), y_true_feat.max()], [y_true_feat.min(), y_true_feat.max()], 'r--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Predictions vs True Values for {col_name} (R² = {r2:.3f})')
            save_path = self.save_path / Path("pred_vs_true")
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / Path(f"{col_name}.png"), dpi=150)
            plt.close()


    @staticmethod
    def _plot_loss_series(ax, values, label, color) -> None:
        """Plot a loss series and its centered five-epoch rolling mean.

        Args:
            ax: Matplotlib axes receiving the series.
            values: Loss values ordered by epoch.
            label: Legend label for the series.
            color: Line color.

        Returns:
            None: Adds the series to the axes.
        """
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return
        epochs = np.arange(len(values))
        smoothed = pd.Series(values).rolling(window=5, center=True, min_periods=1).mean()
        ax.plot(epochs, smoothed, label=label, color=color, linestyle='-', linewidth=1.5)

    def plot_loss_curves(self) -> None:
        """Plot smoothed per-target training and validation losses.

        Args:
            None: Uses this evaluator's target names and model directory.

        Returns:
            None: Saves a combined per-target loss plot.
        """

        loss_load_path = self.file_manager.paths.model_directory(
            self.config_name,
            self.model_save_name,
        )
        train_losses_per_var = self._load_ordered_losses(loss_load_path, 'Train')
        val_losses_per_var = self._load_ordered_losses(loss_load_path, 'Val')
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, name in enumerate(self.target_cols):
            self._plot_loss_series(ax, train_losses_per_var[i], f'{name} (Train loss)', 'blue')
            self._plot_loss_series(ax, val_losses_per_var[i], f'{name} (Validation loss)', 'red')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training & Validation Loss (5-epoch average)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=2)

        plt.tight_layout()
        save_path = self.save_path / Path("loss_curves")
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"loss_curve.png", dpi=150)
        plt.close()

    def plot_raw_loss_curves(self) -> None:
        """Plot the raw mean training and validation losses across targets.

        Args:
            None: Uses this evaluator's model directory.

        Returns:
            None: Saves a plot of the training and validation losses.
        """
        loss_load_path = self.file_manager.paths.model_directory(
            self.config_name,
            self.model_save_name,
        )
        fig, ax = plt.subplots(figsize=(12, 7))

        for split, color in [('Train', 'blue'), ('Val', 'red')]:
            losses = np.load(loss_load_path / f"{split.lower()}_losses_per_var.npy", allow_pickle=True)
            if losses.size == 0:
                continue
            total_loss = np.mean(losses, axis=0)
            epochs = np.arange(len(total_loss))
            ax.plot(epochs, total_loss, label=split, color=color, linewidth=1.5)
            min_loss = np.min(total_loss)
            ax.axhline(min_loss, color=color, alpha=0.6, linewidth=1, linestyle='--',
                       label=f'{split} min: {min_loss:.4g}')


        ax.set_ylim(bottom=4e-4)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean loss across all targets (MSE)', fontsize=12)
        ax.set_title('Total Training & Validation Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        save_path = self.save_path / Path("loss_curves")
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "raw_total_loss_curve.png", dpi=150)
        plt.close()


    def plot_single_loss_curves(self) -> None:
        """Plot smoothed training and validation losses in separate target plots.

        Args:
            None: Uses this evaluator's targets and model directory.

        Returns:
            None: Saves a separate loss plot for each target.
        """

        loss_load_path = self.file_manager.paths.model_directory(
            self.config_name,
            self.model_save_name,
        )
        train_losses_per_var = self._load_ordered_losses(loss_load_path, 'Train')
        val_losses_per_var = self._load_ordered_losses(loss_load_path, 'Val')

        for i, name in enumerate(self.target_cols):
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_loss_series(ax, train_losses_per_var[i], 'Train', 'blue')
            self._plot_loss_series(ax, val_losses_per_var[i], 'Val', 'red')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (MSE)', fontsize=12)
            ax.set_title(f'Training & Validation Loss for {name} (5-epoch average)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            save_path = self.save_path / Path("loss_curves")
            save_path.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path / Path(f"{name}_loss_curve.png"), dpi=150)
            plt.close()

    def plot_total_loss_curve(self) -> None:
        """Plot mean loss across targets and mark each split's minimum.

        Args:
            None: Uses this evaluator's model directory.

        Returns:
            None: Saves the combined total-loss plot.
        """
        loss_load_path = self.file_manager.paths.model_directory(
            self.config_name,
            self.model_save_name,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        for split, color in [('Train', 'blue'), ('Val', 'red')]:
            losses = np.load(loss_load_path / f"{split.lower()}_losses_per_var.npy", allow_pickle=True)
            if losses.size == 0:
                continue
            total_loss = np.mean(losses, axis=0)
            self._plot_loss_series(ax, total_loss, split, color)
            min_loss = np.min(total_loss)
            ax.axhline(min_loss, color=color, alpha=0.6, linewidth=1, linestyle='--',
                       label=f'{split} min: {min_loss:.4g}')

        ax.set_ylim(bottom=4e-4)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean loss across all targets (MSE)', fontsize=12)
        ax.set_title('Total Training & Validation Loss (5-epoch average)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        save_path = self.save_path / "loss_curves"
        save_path.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path / "total_loss_curve.png", dpi=150)
        plt.close(fig)


    def one_v_one_scatter_plot(self) -> None:
        """Plot each target against every other target's prediction error.

        Args:
            None: Uses the prediction and target arrays stored on this instance.

        Returns:
            None: Saves the target-pair scatter plots.
        """

        y_true_dict = {col: self.y_true[:, idx] for idx, col in enumerate(self.target_cols)}
        y_pred_dict = {col: self.y_pred[:, idx] for idx, col in enumerate(self.target_cols)}

        n = len(self.target_cols)

        # Create figure with GridSpec: n rows, n columns
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(n, n, hspace=0.3, wspace=0.3)

        error_list = [y_pred_dict[col] - y_true_dict[col] for col in self.target_cols]

        # Create subdirectory for individual plots
        individual_plots_dir = self.save_path / "one_v_one_scatter"
        individual_plots_dir.mkdir(parents=True, exist_ok=True)

        # Create scatter plots
        for row in range(n):
            for col in range(n):
                ax = fig.add_subplot(gs[row, col])

                # Create DataFrame for seaborn visualization
                true_vals = y_true_dict[self.target_cols[row]]
                error_vals = error_list[col]

                df = pd.DataFrame({
                    'true_value': true_vals,
                    'error': error_vals,
                    'error_bin': pd.qcut(error_vals, q=1, duplicates='drop')
                })

                row_label = display_parameter_name(self.target_cols[row])
                column_label = display_parameter_name(self.target_cols[col])
                ax.set_title(f'{row_label} vs {column_label}-Error', fontsize=9)
                ax.set_xlabel(row_label, fontsize=8)
                ax.set_ylabel(f'{column_label} Error', fontsize=8)

                sns.scatterplot(
                    data=df,
                    x='true_value',
                    y='error',
                    hue='error_bin',
                    ax=ax,
                    alpha=0.6,
                    marker="x",
                    s=30,
                    palette='viridis',
                )

                # Remove legend
                if ax.get_legend():
                    ax.get_legend().remove()

                # Center y-axis at 0
                y_max = np.abs(error_vals).max() * 1.1
                ax.set_ylim(-y_max, y_max)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.tick_params(labelsize=7)

                x_min = true_vals.min() - (true_vals.max() - true_vals.min()) * 0.05
                x_max = true_vals.max() + (true_vals.max() - true_vals.min()) * 0.05

                # Save individual plot with joint plot (includes marginal histograms)
                g = sns.jointplot(
                    data=df,
                    x='true_value',
                    y='error',
                    xlim=(x_min, x_max),
                    hue='error_bin',
                    kind='scatter',
                    height=8,
                    alpha=0.6,
                    marker="x",
                    palette='viridis'
                )
                g.ax_marg_x.set_visible(False)
                g.ax_joint.margins(0.0)


                g.fig.suptitle(f'{row_label} vs {column_label}-Error', fontsize=12)
                g.ax_joint.set_ylim(-y_max, y_max)
                g.ax_joint.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

                # Remove legend
                if g.ax_joint.get_legend():
                    g.ax_joint.get_legend().remove()

                plot_filename = f"{self.target_cols[row]}_vs_{self.target_cols[col]}_error.png"
                g.savefig(individual_plots_dir / plot_filename, dpi=150, bbox_inches='tight')
                plt.close(g.fig)

        plt.savefig(self.save_path / f"one_v_one_scatter.png", dpi=150)
        plt.close()


    def mae_all_combinations_plot(self) -> None:
        """Plot mean absolute errors across target-value combinations.

        Args:
            None: Uses the prediction and target arrays stored on this instance.

        Returns:
            None: Saves the target-pair MAE plots.
        """

        y_true_dict = {col: self.y_true[:, idx] for idx, col in enumerate(self.target_cols)}
        y_pred_dict = {col: self.y_pred[:, idx] for idx, col in enumerate(self.target_cols)}

        fig, axs = plt.subplots(len(self.target_cols), len(self.target_cols), figsize=(15, 15))

        error_list = [y_pred_dict[col] - y_true_dict[col] for col in self.target_cols]

        idx = 0
        for ax in axs.flat:
            idx_col = idx % len(self.target_cols)
            idx_row = idx // len(self.target_cols)

            df = pd.DataFrame({
                "true": y_true_dict[self.target_cols[idx_row]],
                "abs_error": np.abs(error_list[idx_col])
            })

            df["bin"] = pd.qcut(df["true"], q=20, duplicates='drop')
            grouped = df.groupby("bin")["abs_error"]
            mae_per_bin = grouped.mean()
            std_per_bin = grouped.std().fillna(0)

            x_vals = np.array([interval.mid for interval in mae_per_bin.index])
            y_vals = mae_per_bin.values
            std_vals = std_per_bin.values

            # Create DataFrame for seaborn
            plot_df = pd.DataFrame({
                "True Value": x_vals,
                "MAE": y_vals,
                "std": std_vals
            })

            # Plot line using seaborn
            sns.lineplot(data=plot_df, x="True Value", y="MAE", ax=ax,
                        color="steelblue", linewidth=2.5, label="MAE")

            # Add shaded area for standard deviation
            ax.fill_between(x_vals, y_vals - std_vals, y_vals + std_vals,
                           alpha=0.3, color="steelblue", label="±1 Std Dev")

            row_label = display_parameter_name(self.target_cols[idx_row])
            column_label = display_parameter_name(self.target_cols[idx_col])
            ax.set_title(f"{row_label} vs {column_label}-Error")
            ax.set_xlabel(f"True Value ({row_label})")
            ax.set_ylabel(f"MAE ({column_label})")
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            idx += 1

        plt.suptitle("MAE vs True Values for All Combinations", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.save_path / f"mae_all_combinations.png", dpi=150)
        plt.close()


    def good_bad_target_combinations_plot(self) -> None:
        """Plot mean prediction error for every pair of test-set targets.

        Args:
            None: Uses the test-set configuration and stored predictions.

        Returns:
            None: Saves a heatmap of prediction errors across target pairs.

        Raise:
            ValueError: If predictions, targets, or target values are invalid.
        """

        test_config = self.file_manager.load_config(self.test_set_name)
        parameter_sets = test_config.get("parameter_sets", {})
        sim_names = sorted(parameter_sets)
        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch between predictions and targets: {y_pred.shape} != {y_true.shape}"
            )
        if y_true.ndim != 2:
            raise ValueError(f"Expected two-dimensional targets, got shape {y_true.shape}")
        if len(sim_names) != len(y_true):
            raise ValueError(
                f"Number of simulations does not match predictions: {len(sim_names)} != {len(y_true)}"
            )

        prediction_errors = np.abs(y_pred - y_true)
        if not np.isfinite(prediction_errors).all():
            raise ValueError("Prediction errors contain non-finite values")
        average_errors = prediction_errors.mean(axis=1)

        target_pairs = list(combinations(self.target_cols, 2))
        output_path = self.save_path / "good_bad_target_combinations.png"
        if not target_pairs:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "At least two predicted targets are required",
                ha="center",
                va="center",
            )
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return

        values_by_target = {}
        axis_by_target = {}
        configured_ranges = test_config.get("parameter_ranges", {})
        max_axis_values = 25
        for target_index, target in enumerate(self.target_cols):
            values = np.asarray(y_true[:, target_index], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Target '{target}' contains non-finite test-set values")

            unique_values = np.unique(values)
            configured_range = configured_ranges.get(target)
            if configured_range is None:
                lower, upper = float(values.min()), float(values.max())
            else:
                lower, upper = map(float, configured_range[:2])
                lower = min(lower, float(values.min()))
                upper = max(upper, float(values.max()))
            if lower > upper:
                lower, upper = upper, lower
            if lower == upper:
                lower -= 0.5
                upper += 0.5

            if len(unique_values) > max_axis_values:
                axis_values = np.linspace(lower, upper, max_axis_values + 1)
                is_binned = True
            else:
                axis_values = unique_values
                is_binned = False
            values_by_target[target] = values
            axis_by_target[target] = (axis_values, lower, upper, is_binned)

        if len(target_pairs) == 3:
            n_rows, n_columns = 1, 3
        elif len(target_pairs) == 4:
            n_rows, n_columns = 2, 2
        else:
            n_columns = min(3, len(target_pairs))
            n_rows = int(np.ceil(len(target_pairs) / n_columns))
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(5 * n_columns, 5 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        axes = list(axes.flat)
        cmap = plt.get_cmap("RdBu").reversed()
        for axis, (x_target, y_target) in zip(axes, target_pairs):
            x_axis_values, x_lower, x_upper, x_is_binned = axis_by_target[x_target]
            y_axis_values, y_lower, y_upper, y_is_binned = axis_by_target[y_target]
            if x_is_binned:
                x_indices = np.digitize(
                    values_by_target[x_target], x_axis_values[1:-1]
                )
                x_cell_count = len(x_axis_values) - 1
            else:
                x_indices = np.searchsorted(
                    x_axis_values, values_by_target[x_target]
                )
                x_cell_count = len(x_axis_values)
            if y_is_binned:
                y_indices = np.digitize(
                    values_by_target[y_target], y_axis_values[1:-1]
                )
                y_cell_count = len(y_axis_values) - 1
            else:
                y_indices = np.searchsorted(
                    y_axis_values, values_by_target[y_target]
                )
                y_cell_count = len(y_axis_values)
            error_sum = np.zeros(
                (y_cell_count, x_cell_count),
                dtype=float,
            )
            sample_count = np.zeros_like(error_sum)
            np.add.at(
                error_sum,
                (y_indices, x_indices),
                average_errors,
            )
            np.add.at(
                sample_count,
                (y_indices, x_indices),
                1,
            )
            error_grid = np.full_like(error_sum, np.nan)
            np.divide(
                error_sum,
                sample_count,
                out=error_grid,
                where=sample_count > 0,
            )
            finite_errors = error_grid[np.isfinite(error_grid)]
            error_min = float(finite_errors.min())
            error_max = float(finite_errors.max())
            if error_min == error_max:
                padding = max(abs(error_min) * 0.05, 1e-12)
                error_min = max(0.0, error_min - padding)
                error_max += padding
            norm = Normalize(vmin=error_min, vmax=error_max)

            image = axis.imshow(
                np.ma.masked_invalid(error_grid),
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                extent=(x_lower, x_upper, y_lower, y_upper),
                cmap=cmap,
                norm=norm,
            )
            axis.set_box_aspect(1)
            x_label = display_parameter_name(x_target)
            y_label = display_parameter_name(y_target)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(f"{x_label} vs {y_label} error")
            colorbar = fig.colorbar(
                image,
                ax=axis,
                shrink=0.9,
            )
            colorbar.set_label("Mean absolute error")

        for axis in axes[len(target_pairs):]:
            axis.axis("off")

        fig.suptitle("Prediction error across target combinations")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    def abs_error_histograms(self, bins: int = 50) -> None:
        """Plot absolute error histograms for all targets into one image.

        Args:
            bins: Number of histogram bins.

        Returns:
            None: Saves a combined absolute-error histogram.
        """

        errors = np.abs(self.y_pred - self.y_true)
        n_targets = len(self.target_cols)
        n_cols = n_targets
        n_rows = int(np.ceil(n_targets / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, col_name in enumerate(self.target_cols):
            ax = axes[idx]
            col_errors = errors[:, idx]
            average_error = np.mean(col_errors)
            threshold = np.percentile(col_errors, 90)
            bin_edges = np.histogram_bin_edges(col_errors, bins=bins)

            low = col_errors[col_errors <= threshold]
            high = col_errors[col_errors > threshold]

            ax.hist(low, bins=bin_edges, color="steelblue", alpha=0.8, edgecolor="black", label="Bottom 90%")
            ax.hist(high, bins=bin_edges, color="tomato", alpha=0.8, edgecolor="black", label="Top 10%")
            ax.axvline(threshold, color="black", linestyle="-", linewidth=1.2, label=f"p90 = {threshold:.4f}")
            ax.axvline(average_error, color="black", linestyle="--", linewidth=1.2,
                       label=f"Average Error = {average_error:.4f}")
            ax.set_xlabel("Absolute Error")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Absolute Error Histogram – {col_name}")
            ax.legend(fontsize=8)

        for ax in axes[n_targets:]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.save_path / Path("abs_error_hist_all.png"), dpi=150)
        plt.close()


