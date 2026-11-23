"""Helpers for grouping predictions by range-normalized error."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from spin_system import FileManager
from .evaluator import Evaluator


def range_normalized_absolute_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: Sequence[str],
    parameter_ranges: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, list[str]]:
    """Return absolute errors normalized by each parameter's range width.

    Targets with a zero-width configured range are omitted because their
    prediction errors cannot be normalized meaningfully.

    Args:
        y_true: Ground-truth target matrix.
        y_pred: Predicted target matrix with the same shape as ``y_true``.
        target_cols: Target names corresponding to matrix columns.
        parameter_ranges: Configured lower and upper bounds for each target.

    Returns:
        tuple[np.ndarray, list[str]]: Normalized errors and names of included
        targets.

    Raise:
        ValueError: If shapes, values, or configured ranges are invalid.
    """
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "Shape mismatch between predictions and targets: "
            f"{predicted_values.shape} != {true_values.shape}"
        )
    if true_values.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional predictions and targets, got shape {true_values.shape}"
        )
    if true_values.shape[1] != len(target_cols):
        raise ValueError(
            "Number of target columns does not match prediction columns: "
            f"{len(target_cols)} != {true_values.shape[1]}"
        )
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Predictions and targets must contain only finite values")

    normalized_errors = []
    parameter_targets = []
    for index, target in enumerate(target_cols):
        if target not in parameter_ranges:
            raise ValueError(f"Missing configured range for target '{target}'")
        parameter_range = parameter_ranges[target]
        if len(parameter_range) < 2:
            raise ValueError(f"Configured range for target '{target}' requires two values")

        lower, upper = map(float, parameter_range[:2])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"Configured range for target '{target}' must be finite")
        range_width = abs(upper - lower)
        if range_width == 0:
            continue

        normalized_errors.append(
            np.abs(predicted_values[:, index] - true_values[:, index]) / range_width
        )
        parameter_targets.append(target)

    if not normalized_errors:
        raise ValueError("At least one target must have a non-zero configured range")

    return np.column_stack(normalized_errors), parameter_targets


def split_indices_by_normalized_errors(
    normalized_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split samples into the best 90% and the worst 10% overall.

    Args:
        normalized_errors: Per-sample, per-target normalized error matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Indices for the best and worst subsets.

    Raise:
        ValueError: If the error matrix is empty, malformed, or non-finite.
    """
    errors = np.asarray(normalized_errors, dtype=float)
    if errors.ndim != 2 or errors.shape[1] == 0:
        raise ValueError(
            "Expected normalized errors with shape (n_samples, n_targets)"
        )
    if errors.shape[0] == 0:
        raise ValueError("Cannot split an empty set of normalized errors")
    if not np.isfinite(errors).all():
        raise ValueError("Normalized errors must contain only finite values")

    mean_errors = errors.mean(axis=1)
    threshold = np.percentile(mean_errors, 90)
    return np.flatnonzero(mean_errors <= threshold), np.flatnonzero(mean_errors > threshold)


def split_prediction_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: Sequence[str],
    parameter_ranges: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Split predictions using their mean range-normalized parameter error.

    Args:
        y_true: Ground-truth target matrix.
        y_pred: Predicted target matrix.
        target_cols: Target names corresponding to matrix columns.
        parameter_ranges: Configured lower and upper bounds for each target.

    Returns:
        tuple[np.ndarray, np.ndarray]: Indices for the best and worst subsets.
    """
    normalized_errors, _ = range_normalized_absolute_errors(
        y_true,
        y_pred,
        target_cols,
        parameter_ranges,
    )
    return split_indices_by_normalized_errors(normalized_errors)


def filter_predictions(
    evaluator: Evaluator,
    test_set_name: str,
    model_save_name: str,
    output_dir: Path | None = None,
    file_manager: FileManager | None = None,
) -> tuple[Path, Path]:
    """Save simulation names split by mean range-normalized parameter error.

    Args:
        evaluator: Evaluator with aligned predictions, targets, and target names.
        test_set_name: Name of the test-set configuration.
        model_save_name: Model checkpoint name used to name output files.
        output_dir: Optional directory for the two selection files.
        file_manager: Optional manager for configuration and output paths.

    Returns:
        tuple[Path, Path]: Paths to the good-sample and bad-sample arrays.

    Raise:
        ValueError: If simulation counts do not match or errors cannot be split.
    """
    y_true = np.asarray(evaluator.y_true)
    y_pred = np.asarray(evaluator.y_pred)

    file_manager = file_manager or FileManager()
    test_config = file_manager.load_config(test_set_name)
    good_indices, bad_indices = split_prediction_indices(
        y_true,
        y_pred,
        evaluator.target_cols,
        test_config.get("parameter_ranges", {}),
    )
    sim_names = np.asarray(sorted(test_config["parameter_sets"].keys()))
    if len(sim_names) != len(y_true):
        raise ValueError(
            f"Number of simulations does not match predictions: {len(sim_names)} != {len(y_true)}"
        )
    if output_dir is None:
        output_dir = file_manager.paths.config_directory(test_set_name)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_save_name).stem
    good_path = output_dir / f"{model_name}_good.npy"
    bad_path = output_dir / f"{model_name}_bad.npy"
    np.save(good_path, sim_names[good_indices])
    np.save(bad_path, sim_names[bad_indices])

    print(f"Saved {len(good_indices)} good samples to {good_path}")
    print(f"Saved {len(bad_indices)} bad samples to {bad_path}")
    return good_path, bad_path
