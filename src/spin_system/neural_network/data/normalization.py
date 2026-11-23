"""Normalize simulation data sources before neural-network input."""

import numpy as np


def normalize_input_array(array: np.ndarray, source: str) -> np.ndarray:
    """Scale magnitude arrays to [-1, 1] and leave other sources unchanged.

    Args:
        array: Input simulation data.
        source: Data-source name that determines whether scaling is applied.

    Returns:
        np.ndarray: Normalized magnitude data or the unchanged input array.

    Raise:
        ValueError: If magnitude data contains non-finite values.
    """
    values = np.asarray(array)
    if source != "magnitude":
        return values

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Magnitude data must contain only finite values")

    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.float32)

    normalized = 2 * (values - minimum) / (maximum - minimum) - 1
    return normalized.astype(np.float32)
