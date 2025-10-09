"""
Reusable utility helpers for the COSMIC package.

The original repository only exposed the :class:`CosmicAnalyzer` class.  To
keep the main module lean we collect small, composable helpers here so they can
be reused across notebooks or custom analysis scripts.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "median_absolute_deviation",
    "mad_z_score",
    "detect_outliers",
    "normalize_columns",
    "ensure_columns",
    "linear_gradient",
    "tetrahedron_quality_factors",
]


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate that *columns* exist in ``df``."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def median_absolute_deviation(values: ArrayLike, *, scale: float = 1.4826) -> float:
    """
    Compute the (scaled) Median Absolute Deviation (MAD).

    Parameters
    ----------
    values:
        Iterable of observations.
    scale:
        Scaling factor applied to match the standard deviation of a normal
        distribution (default 1.4826).
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan")

    median = np.nanmedian(data)
    deviations = np.abs(data - median)
    mad = scale * np.nanmedian(deviations)
    return float(mad)


def mad_z_score(values: ArrayLike, *, scale: float = 1.4826) -> np.ndarray:
    """
    Robust Z-score based on the Median Absolute Deviation.

    Returns an array whose absolute value represents how extreme each element
    is relative to the distribution's median.
    """
    data = np.asarray(values, dtype=float)
    mad = median_absolute_deviation(data, scale=scale)
    if np.isnan(mad) or mad == 0.0:
        return np.zeros_like(data, dtype=float)
    median = np.nanmedian(data)
    return np.abs(data - median) / mad


def detect_outliers(
    series: pd.Series | ArrayLike,
    *,
    threshold: float = 3.5,
    robust: bool = True,
) -> pd.Series:
    """
    Identify outliers using either a robust MAD score or standard deviation.

    Parameters
    ----------
    series:
        Input data.  The function accepts a pandas Series or any array-like
        structure.
    threshold:
        Minimum Z-score/MAD-score considered an outlier.
    robust:
        When ``True`` (default) the routine uses the MAD score; otherwise a
        classical Z-score is applied.
    """
    data = pd.Series(series, dtype=float)
    if data.empty:
        return pd.Series(dtype=bool)

    if robust:
        scores = mad_z_score(data.to_numpy())
    else:
        mean = data.mean()
        std = data.std(ddof=0)
        if std == 0.0 or np.isnan(std):
            scores = np.zeros(len(data))
        else:
            scores = np.abs(data - mean) / std

    return pd.Series(scores > threshold, index=data.index)


def normalize_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    center: bool = True,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` where the selected columns are normalised.

    Parameters
    ----------
    columns:
        Columns to process.  When ``None`` all numeric columns are considered.
    center:
        Subtract the mean before scaling.
    scale:
        Divide by the standard deviation (if non-zero).
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    ensure_columns(df, columns)

    normalised = df.copy()
    for column in columns:
        series = normalised[column].astype(float)
        if center:
            series = series - series.mean()
        if scale:
            std = series.std(ddof=0)
            if std != 0.0:
                series = series / std
        normalised[column] = series
    return normalised


def _prepare_positions(positions: ArrayLike) -> np.ndarray:
    coords = np.asarray(positions, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Positions must be an array of shape (N, 3).")
    return coords


def linear_gradient(
    positions: ArrayLike,
    values: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a first-order (affine) model to data measured at multiple positions.

    Parameters
    ----------
    positions:
        Coordinates with shape ``(N, 3)``.
    values:
        Scalar or vector quantity with shape ``(N,)`` or ``(N, M)``.

    Returns
    -------
    gradient:
        Array with shape ``(M, 3)`` containing the spatial gradients.  For scalar
        inputs ``M = 1``.
    intercept:
        Values of the fitted function at the centroid of the input positions.
    """
    coords = _prepare_positions(positions)
    measurements = np.asarray(values, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements[:, np.newaxis]

    if measurements.shape[0] != coords.shape[0]:
        raise ValueError("`positions` and `values` must share the first dimension.")

    offsets = coords - coords.mean(axis=0)
    design = np.hstack((np.ones((offsets.shape[0], 1)), offsets))

    gradients = np.zeros((measurements.shape[1], 3), dtype=float)
    intercepts = np.zeros(measurements.shape[1], dtype=float)

    for idx in range(measurements.shape[1]):
        coeffs, *_ = np.linalg.lstsq(design, measurements[:, idx], rcond=None)
        intercepts[idx] = coeffs[0]
        gradients[idx] = coeffs[1:]

    return gradients, intercepts


def tetrahedron_quality_factors(positions: ArrayLike) -> Tuple[float, float, float]:
    """
    Compute diagnostic quality factors for a tetrahedral configuration.

    The implementation follows the spirit of the ``Q_G`` (Glassmeier) and
    ``Q_R`` (Robert-Roux) factors presented by Daly (1994), expressed through the
    eigenvalues of the position covariance matrix.  ``Q_G`` ranges from 1
    (colinear) to 3 (regular tetrahedron), while ``Q_R`` ranges from 0 to 1 and
    highlights volumetric quality.

    Returns
    -------
    q_g, q_r, volume:
        The two dimensionless quality factors and the tetrahedron volume in the
        same units as the input coordinates cubed.
    """
    coords = _prepare_positions(positions)
    if coords.shape[0] != 4:
        raise ValueError("Exactly four positions are required for the tetrahedron factors.")

    centroid = coords.mean(axis=0)
    offsets = coords - centroid
    moment = offsets.T @ offsets / offsets.shape[0]

    eigenvalues = np.sort(np.linalg.eigvalsh(moment))[::-1]
    if np.allclose(eigenvalues, 0.0):
        q_g = 0.0
        q_r = 0.0
    else:
        lambda_sum = np.sum(eigenvalues)
        lambda_sq_sum = np.sum(eigenvalues**2)
        q_g = (lambda_sum**2) / lambda_sq_sum if lambda_sq_sum != 0.0 else 0.0

        product = np.prod(eigenvalues)
        if product <= 0.0 or lambda_sum == 0.0:
            q_r = 0.0
        else:
            q_r = (3.0 * product ** (1.0 / 3.0)) / lambda_sum

    base = coords[1:] - coords[0]
    volume = abs(np.linalg.det(base)) / 6.0

    return float(q_g), float(q_r), float(volume)
