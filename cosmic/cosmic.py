"""
Core analysis utilities for the COSMIC project.

This module exposes :class:`CosmicAnalyzer`, a battery of routines used to
process magnetometer data from the ESA Cluster mission.  The original project
started life inside notebooks; the code below reshapes those experiments into a
clean, reusable, and well documented Python API ready for production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.stats import genpareto

from .utils import (
    detect_outliers,
    ensure_columns,
    linear_gradient,
    normalize_columns,
    tetrahedron_quality_factors,
)

MU_0: float = 1.25663706212e-6  # Permeability of Free Space (H/m)
# Backwards compatibility with the previous lowercase constant
mu_0: float = MU_0

DEFAULT_POSITION_COLUMNS: Tuple[str, str, str] = (
    "sc_pos_xyz_gse__C1_CP_FGM_FULL1",
    "sc_pos_xyz_gse__C1_CP_FGM_FULL2",
    "sc_pos_xyz_gse__C1_CP_FGM_FULL3",
)

DEFAULT_MAGNETIC_COLUMNS: Tuple[str, str, str] = (
    "B_vec_xyz_gse__C1_CP_FGM_FULL1",
    "B_vec_xyz_gse__C1_CP_FGM_FULL2",
    "B_vec_xyz_gse__C1_CP_FGM_FULL3",
)

FF_THRESHOLD: float = 0.15  # Fraction of angles above threshold needed for detection

__all__ = [
    "CosmicAnalyzer",
    "cosmic",
    "MU_0",
    "mu_0",
    "DEFAULT_POSITION_COLUMNS",
    "DEFAULT_MAGNETIC_COLUMNS",
    "SpectrumResult",
    "MVAResult",
    "TimingResult",
]


@dataclass(frozen=True)
class MagneticConfig:
    """Configuration describing which columns hold position and magnetic data."""

    position_columns: Sequence[str] = DEFAULT_POSITION_COLUMNS
    magnetic_columns: Sequence[str] = DEFAULT_MAGNETIC_COLUMNS

    def validate(self, df: pd.DataFrame) -> None:
        """Ensure the target columns are present in the dataframe."""
        missing = [
            column
            for column in (*self.position_columns, *self.magnetic_columns)
            if column not in df.columns
        ]
        if missing:
            raise KeyError(f"Required columns are missing from dataframe: {missing}")


@dataclass(frozen=True)
class SpectrumResult:
    """Container for spectral analysis results."""

    frequencies_hz: np.ndarray
    psd: np.ndarray
    slope: float | None = None
    slope_range: Tuple[float, float] | None = None


@dataclass(frozen=True)
class MVAResult:
    """Outputs from the minimum variance analysis."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    elongation_ratio: float
    planarity_ratio: float


@dataclass(frozen=True)
class TimingResult:
    """Normal vector and velocity derived from multi-spacecraft timing."""

    normal: np.ndarray
    velocity: float
    residual: float


class CosmicAnalyzer:
    """
    Main entry-point for magnetometer analysis routines.

    Parameters
    ----------
    config:
        Optional magnetic configuration.  By default the class uses the column
        names produced by the Cluster mission datasets.
    matplotlib_backend_safe:
        When ``True`` (default) plotting methods will use ``plt.show``.  Set to
        ``False`` in automated environments to skip accidental GUI calls.
    """

    def __init__(
        self,
        config: MagneticConfig | None = None,
        *,
        matplotlib_backend_safe: bool = True,
    ) -> None:
        self.config = config or MagneticConfig()
        self._matplotlib_backend_safe = matplotlib_backend_safe

    # ------------------------------------------------------------------ #
    # I/O utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def ler_arquivo_dat(file_path: str) -> pd.DataFrame:
        """
        Parse a CEF-like ``.dat`` file into a :class:`pandas.DataFrame`.

        The format stores metadata lines such as ``START_VARIABLE`` and
        ``SIZES``; data rows are comma separated.  The function builds column
        names dynamically based on the metadata definitions.
        """

        variables: List[str] = []
        rows: List[List[str]] = []
        current_variable: Optional[str] = None

        with open(file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("EOF"):
                    break

                if "START_VARIABLE" in line:
                    current_variable = line.split("=", maxsplit=1)[1].strip()
                    continue

                if "SIZES" in line:
                    if current_variable is None:
                        raise ValueError(
                            "Encountered SIZES declaration before START_VARIABLE."
                        )
                    count = int(line.split("=", maxsplit=1)[1])
                    variables.extend(f"{current_variable}{idx}" for idx in range(1, count + 1))
                    continue

                parts = [part.strip() for part in line.split(",")]
                # Only keep rows that match the expected number of variables; this
                # filters out stray metadata lines and blank entries.
                if variables and len(parts) == len(variables):
                    rows.append(parts)

        if not variables:
            raise ValueError(
                f"Unable to infer variables from file '{file_path}'. "
                "Please double-check the input."
            )

        frame = pd.DataFrame(rows, columns=variables)
        return frame.apply(pd.to_numeric, errors="ignore")

    # ------------------------------------------------------------------ #
    # Vector algebra helpers
    # ------------------------------------------------------------------ #

    def _magnetics(self, df: pd.DataFrame) -> np.ndarray:
        self.config.validate(df)
        return df[list(self.config.magnetic_columns)].astype(float).to_numpy()

    def _positions(self, df: pd.DataFrame) -> np.ndarray:
        self.config.validate(df)
        return df[list(self.config.position_columns)].astype(float).to_numpy()

    @staticmethod
    def _norm(vector: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.linalg.norm(vector, axis=axis)

    @staticmethod
    def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

    # ------------------------------------------------------------------ #
    # Core analyses
    # ------------------------------------------------------------------ #

    def calculate_B_diff(self, df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
        """Vector difference between the magnetic fields of two spacecraft."""
        return self._magnetics(df1) - self._magnetics(df2)

    def calculate_r_diff(self, df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
        """Vector difference between the positions of two spacecraft."""
        return self._positions(df1) - self._positions(df2)

    def calculate_current_density(
        self, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame
    ) -> pd.Series:
        """
        Compute the current density between three spacecraft using the curlometer
        formulation (eq. 2 in Dunlop et al., 2002).
        """
        r13 = self.calculate_r_diff(df1, df3)
        r23 = self.calculate_r_diff(df2, df3)

        B13 = self.calculate_B_diff(df1, df3)
        B23 = self.calculate_B_diff(df2, df3)

        cross_product = np.cross(r13, r23)
        denominator = np.einsum("ij,ij->i", cross_product, cross_product)
        if np.any(denominator == 0.0):
            raise ZeroDivisionError(
                "Degenerate geometry detected: spacecraft vectors are co-linear."
            )

        numerator = np.einsum("ij,ij->i", B13, r23) - np.einsum("ij,ij->i", B23, r13)
        current_density = (1.0 / MU_0) * numerator / denominator
        return pd.Series(current_density)

    def curlometer(
        self,
        spacecraft1: pd.DataFrame,
        spacecraft2: pd.DataFrame,
        spacecraft3: pd.DataFrame,
        spacecraft4: pd.DataFrame,
    ) -> pd.Series:
        """Average the current densities from the four possible tetrahedra."""
        components = [
            self.calculate_current_density(spacecraft1, spacecraft2, spacecraft3),
            self.calculate_current_density(spacecraft1, spacecraft2, spacecraft4),
            self.calculate_current_density(spacecraft1, spacecraft3, spacecraft4),
            self.calculate_current_density(spacecraft2, spacecraft3, spacecraft4),
        ]
        stacked = pd.concat(components, axis=1)
        # Mirror the behaviour of the original implementation: the result is the
        # magnitude of the average current density.
        return stacked.mean(axis=1).abs()

    def calculate_mod_B(
        self,
        df: pd.DataFrame,
        Bx_column: str,
        By_column: str,
        Bz_column: str,
    ) -> pd.Series:
        """Compute the magnitude of the magnetic field."""
        vectors = df[[Bx_column, By_column, Bz_column]].astype(float).to_numpy()
        return pd.Series(np.linalg.norm(vectors, axis=1))

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #

    def plot_mod_B(self, values: ArrayLike, *, ax: plt.Axes | None = None) -> None:
        """Plot ``|B|`` as a function of index."""
        axis = ax or plt.gca()
        axis.plot(values)
        axis.set_xlabel("Index")
        axis.set_ylabel("|B|")
        axis.set_title("Magnetic Field Magnitude")
        if ax is None and self._matplotlib_backend_safe:
            plt.show()

    def calculate_PVI(self, series: ArrayLike, tau: int = 66) -> pd.Series:
        """
        Compute the Partial Variance of Increments (PVI).

        Parameters
        ----------
        series:
            Scalar magnetic-field magnitude (typically ``|B|``) samples.
        tau:
            Lag, in samples, between the two points used to form the increment.
        """
        data = pd.Series(series, dtype=float)
        if tau <= 0:
            raise ValueError("`tau` has to be a positive integer.")
        if tau >= len(data):
            raise ValueError("`tau` must be smaller than the number of samples.")

        delta = data.iloc[tau:].to_numpy() - data.iloc[:-tau].to_numpy()
        mean_square = np.mean(np.abs(delta) ** 2)
        if mean_square == 0.0:
            return pd.Series(np.zeros_like(delta))
        return pd.Series(np.abs(delta) / np.sqrt(mean_square))

    def plot_data(self, values: ArrayLike, *, ax: plt.Axes | None = None) -> None:
        """Simple helper to plot a time series."""
        axis = ax or plt.gca()
        axis.plot(values)
        axis.set_xlabel("Index")
        axis.set_ylabel("Value")
        if ax is None and self._matplotlib_backend_safe:
            plt.show()

    def angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Return the angle in degrees between two 3D vectors."""
        uv1 = self._unit_vectors(np.asarray(v1, dtype=float).reshape(1, -1))[0]
        uv2 = self._unit_vectors(np.asarray(v2, dtype=float).reshape(1, -1))[0]
        cosine = np.clip(np.dot(uv1, uv2), -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def cs_detection(self, df: pd.DataFrame, tau: int, theta_c: float) -> int:
        """
        Detect current sheets using the method by Li et al.

        The method compares the angle between two consecutive windows.  If more
        than 15% of the angles exceed ``theta_c`` the function flags a current
        sheet.
        """
        if len(df) < 2 * tau:
            raise ValueError("Dataframe must contain at least 2*tau samples.")

        first_window = df.iloc[0:tau].to_numpy(dtype=float)
        second_window = df.iloc[tau : 2 * tau].to_numpy(dtype=float)

        angles = [
            self.angle(first_window[idx], second_window[idx]) for idx in range(tau)
        ]
        fraction = np.mean(np.array(angles) >= theta_c)
        return int(fraction >= FF_THRESHOLD)

    def limethod(
        self,
        df: pd.DataFrame,
        *,
        theta_c: float = 35.0,
        tau_sec: float = 10.0,
        sample_frequency_hz: float = 22.0,
    ) -> pd.DataFrame:
        """
        Sliding-window implementation of the Li et al. method for current sheet detection.

        Parameters
        ----------
        df:
            Time-indexed dataframe with three magnetic field components.
        theta_c:
            Threshold applied to the angle between windowed vectors.
        tau_sec:
            Window length in seconds.
        sample_frequency_hz:
            Sampling frequency of the dataset.
        """
        if sample_frequency_hz <= 0:
            raise ValueError("Sampling frequency must be positive.")

        tau = int(sample_frequency_hz * tau_sec)
        if tau <= 0:
            raise ValueError("Computed tau must be at least 1.")
        if len(df) <= 2 * tau:
            raise ValueError("Insufficient data for the requested window length.")

        outputs: List[Tuple[pd.Timestamp, int]] = []
        for index in range(tau, len(df) - tau):
            window = df.iloc[index - tau : index + tau]
            detection = self.cs_detection(window, tau, theta_c)
            outputs.append((df.index[index], detection))

        return pd.DataFrame(outputs, columns=["Time", "cs_out"])

    # ------------------------------------------------------------------ #
    # Numeric helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def convert_to_float(value: str) -> float:
        """
        Convert scientific notation in either ``E`` or ``D`` format to floats.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(str(value).replace("D", "E"))

    def calculate_magnetic_volatility(
        self,
        df: pd.DataFrame,
        column: str,
        tau: int = 50,
        window: int | None = None,
        *,
        w: int | None = None,
    ) -> pd.Series:
        """
        Rolling standard deviation of log-returns for the supplied column.
        """
        if window is None and w is None:
            window = 50
        elif window is None:
            window = w
        elif w is not None and window != w:
            raise ValueError("Conflicting values for `window` and `w`.")

        if window is None or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        series = df[column].astype(float)
        shifted = series.shift(periods=tau)
        with np.errstate(divide="ignore"):
            delta = np.log(shifted / series)
        vol = delta.rolling(window=window).std()
        return vol

    @staticmethod
    def apply_gaussian_kernel(values: ArrayLike, sigma: float) -> np.ndarray:
        """Smooth a 1D sequence with a Gaussian kernel."""
        return gaussian_filter1d(np.asarray(values, dtype=float), sigma)

    # ------------------------------------------------------------------ #
    # Data conditioning helpers
    # ------------------------------------------------------------------ #

    def calculate_magnetic_energy_density(
        self,
        df: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.Series:
        """
        Magnetic energy density ``B^2 / (2 * mu_0)`` for the selected components.

        Returns
        -------
        pandas.Series
            Series indexed like ``df`` containing the energy density in J/m³.
        """
        target_columns = columns or self.config.magnetic_columns
        ensure_columns(df, target_columns)
        vectors = df[list(target_columns)].astype(float).to_numpy()
        energy_density = np.sum(vectors**2, axis=1) / (2.0 * MU_0)
        return pd.Series(energy_density, index=df.index, name="magnetic_energy_density")

    def normalize_magnetic_field(
        self,
        df: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        center: bool = True,
        scale: bool = True,
    ) -> pd.DataFrame:
        """
        Return a copy of ``df`` with the magnetic field components normalised.

        Uses :func:`cosmic.utils.normalize_columns` under the hood and preserves
        non-numeric columns.
        """
        target_columns = columns or self.config.magnetic_columns
        return normalize_columns(df, target_columns, center=center, scale=scale)

    def remove_outliers(
        self,
        df: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        threshold: float = 3.5,
        robust: bool = True,
        fill_strategy: str | None = "interpolate",
    ) -> pd.DataFrame:
        """
        Replace outliers in the selected columns by ``NaN`` and optionally fill them.

        Parameters
        ----------
        threshold:
            Minimum MAD/Z-score considered an outlier.
        robust:
            When ``True`` (default) use the MAD score; otherwise the classical
            Z-score is applied.
        fill_strategy:
            ``"interpolate"`` (default) performs a time-aware interpolation,
            ``"median"`` fills with the column median, ``None`` leaves the
            missing values untouched.
        """
        target_columns = columns or self.config.magnetic_columns
        ensure_columns(df, target_columns)

        cleaned = df.copy()
        for column in target_columns:
            original_series = cleaned[column].astype(float)
            mask = detect_outliers(original_series, threshold=threshold, robust=robust)
            if not mask.any():
                continue
            series = original_series.copy()
            series[mask] = np.nan

            if fill_strategy is None:
                cleaned[column] = series
                continue
            if fill_strategy == "interpolate":
                method = "time" if isinstance(cleaned.index, pd.DatetimeIndex) else "linear"
                series = series.interpolate(method=method, limit_direction="both")
            elif fill_strategy == "median":
                series = series.fillna(original_series.median())
            else:
                raise ValueError("Unsupported fill strategy. Use 'interpolate', 'median' or None.")

            cleaned[column] = series

        return cleaned

    def resample_dataframe(
        self,
        df: pd.DataFrame,
        frequency: str,
        *,
        columns: Sequence[str] | None = None,
        agg: str = "mean",
        interpolate: bool = True,
    ) -> pd.DataFrame:
        """
        Resample a time-indexed dataframe at a new frequency.

        Parameters
        ----------
        frequency:
            Target frequency string understood by :meth:`pandas.DataFrame.resample`.
        agg:
            Aggregation method applied after resampling (default ``"mean"``).
        interpolate:
            When ``True`` interpolate missing values using time interpolation.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Dataframe index must be a DatetimeIndex to resample.")

        target_columns = columns or df.columns
        ensure_columns(df, target_columns)

        subset = df[list(target_columns)]
        try:
            aggregator = getattr(subset.resample(frequency), agg)
        except AttributeError as exc:
            raise ValueError(f"Aggregation '{agg}' is not supported by pandas.") from exc

        resampled = aggregator()
        if interpolate:
            resampled = resampled.interpolate(method="time", limit_direction="both")
        return resampled

    # ------------------------------------------------------------------ #
    # Spectral analysis
    # ------------------------------------------------------------------ #

    def power_spectral_density(
        self,
        series: ArrayLike,
        sample_frequency_hz: float,
        *,
        nperseg: int | None = None,
        slope_range: Tuple[float, float] | None = None,
        detrend: str = "constant",
        scaling: str = "density",
        average: str = "mean",
    ) -> SpectrumResult:
        """
        Estimate the magnetic-field power spectral density via Welch's method.

        Parameters
        ----------
        series:
            Scalar time series containing the magnetic field component or
            magnitude of interest.
        sample_frequency_hz:
            Sampling rate of the series.
        nperseg:
            Segment length handed to :func:`scipy.signal.welch`.
        slope_range:
            Optional ``(f_min, f_max)`` range (in Hz) used to fit the spectral
            slope in log-log space.
        """
        data = np.asarray(series, dtype=float)
        if data.ndim != 1:
            raise ValueError("`series` must be one-dimensional.")
        if sample_frequency_hz <= 0:
            raise ValueError("`sample_frequency_hz` must be positive.")

        freqs, psd = welch(
            data,
            fs=sample_frequency_hz,
            nperseg=nperseg,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

        slope = None
        if slope_range is not None:
            f_min, f_max = slope_range
            if f_min <= 0 or f_max <= 0 or f_min >= f_max:
                raise ValueError("`slope_range` must contain positive increasing frequencies.")
            mask = (freqs >= f_min) & (freqs <= f_max) & (psd > 0.0)
            if np.count_nonzero(mask) >= 2:
                x = np.log10(freqs[mask])
                y = np.log10(psd[mask])
                slope, _ = np.polyfit(x, y, deg=1)

        return SpectrumResult(freqs, psd, slope, slope_range)

    def component_power_spectra(
        self,
        df: pd.DataFrame,
        sample_frequency_hz: float,
        *,
        columns: Sequence[str] | None = None,
        nperseg: int | None = None,
        slope_range: Tuple[float, float] | None = None,
    ) -> Dict[str, SpectrumResult]:
        """
        Compute PSDs for the total, parallel, and perpendicular magnetic components.

        The mean magnetic field defines the parallel direction.  The perpendicular
        component is the magnitude of the fluctuations orthogonal to that mean.
        """
        target_columns = columns or self.config.magnetic_columns
        ensure_columns(df, target_columns)

        vectors = df[list(target_columns)].astype(float).to_numpy()
        b_mean = np.mean(vectors, axis=0)
        norm = np.linalg.norm(b_mean)
        if norm == 0.0:
            raise ValueError("Mean magnetic field is zero; cannot define parallel/perpendicular axes.")
        b_hat = b_mean / norm

        magnitude = np.linalg.norm(vectors, axis=1)
        parallel = vectors @ b_hat
        perpendicular = np.linalg.norm(vectors - np.outer(parallel, b_hat), axis=1)

        results = {
            "total": self.power_spectral_density(
                magnitude,
                sample_frequency_hz,
                nperseg=nperseg,
                slope_range=slope_range,
            ),
            "parallel": self.power_spectral_density(
                parallel,
                sample_frequency_hz,
                nperseg=nperseg,
                slope_range=slope_range,
            ),
            "perpendicular": self.power_spectral_density(
                perpendicular,
                sample_frequency_hz,
                nperseg=nperseg,
                slope_range=slope_range,
            ),
        }
        return results

    # ------------------------------------------------------------------ #
    # Correlation analysis
    # ------------------------------------------------------------------ #

    def autocorrelation(
        self,
        series: ArrayLike,
        *,
        max_lag: int | None = None,
        demean: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """
        Autocorrelation of a scalar time series.

        Returns the lags (in samples), the autocorrelation coefficients, and the
        first lag index where the correlation drops below ``exp(-1)`` – a proxy
        for the decorrelation time (``None`` when not reached).
        """
        data = pd.Series(series, dtype=float).dropna()
        if data.empty:
            raise ValueError("`series` must contain at least one finite value.")

        if demean:
            data = data - data.mean()

        max_lag = len(data) - 1 if max_lag is None else min(max_lag, len(data) - 1)
        values = data.to_numpy()
        correlation = np.correlate(values, values, mode="full")
        correlation = correlation[len(values) - 1 : len(values) + max_lag]
        correlation /= correlation[0]

        lags = np.arange(0, correlation.size)
        decorrelation_idx = None
        below = np.where(correlation <= np.exp(-1))[0]
        if below.size:
            decorrelation_idx = int(below[0])

        return lags, correlation, decorrelation_idx

    def cross_correlation(
        self,
        series1: ArrayLike,
        series2: ArrayLike,
        sample_frequency_hz: float,
        *,
        max_lag: int | None = None,
        demean: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Cross-correlation between two scalar series.

        Returns the lag array (in samples), the correlation coefficients and the
        lag in seconds that maximises the absolute correlation.
        """
        s1 = pd.Series(series1, dtype=float).dropna()
        s2 = pd.Series(series2, dtype=float).dropna()
        if s1.empty or s2.empty:
            raise ValueError("Both series must contain finite values.")

        n = min(len(s1), len(s2))
        s1, s2 = s1.iloc[:n], s2.iloc[:n]
        if demean:
            s1 = s1 - s1.mean()
            s2 = s2 - s2.mean()

        values1 = s1.to_numpy()
        values2 = s2.to_numpy()
        max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)

        full_lags = np.arange(-n + 1, n)
        raw_correlation = np.correlate(values1, values2, mode="full")
        denom = np.linalg.norm(values1) * np.linalg.norm(values2)
        if denom == 0.0:
            correlation_full = np.zeros_like(raw_correlation)
        else:
            correlation_full = raw_correlation / denom

        center = len(correlation_full) // 2
        start = center - max_lag
        stop = center + max_lag + 1
        lags = full_lags[start:stop]
        correlation = correlation_full[start:stop]

        best_idx = int(np.argmax(np.abs(correlation)))
        best_lag_seconds = float(lags[best_idx]) / sample_frequency_hz

        return lags, correlation, best_lag_seconds

    # ------------------------------------------------------------------ #
    # Structure functions and higher-order statistics
    # ------------------------------------------------------------------ #

    def structure_functions(
        self,
        series: ArrayLike,
        *,
        orders: Sequence[float] = (2.0,),
        lags: Sequence[int] = (1,),
    ) -> pd.DataFrame:
        """
        Compute the structure function ``S_p(tau) = <|ΔB(tau)|^p>``.
        """
        data = pd.Series(series, dtype=float).dropna()
        if data.empty:
            raise ValueError("`series` must contain finite values.")

        results: List[Tuple[int, float, float]] = []
        values = data.to_numpy()
        n = len(values)

        for tau in lags:
            if tau <= 0 or tau >= n:
                raise ValueError("Each lag must satisfy 0 < tau < len(series).")
            delta = values[tau:] - values[:-tau]
            for order in orders:
                structure_value = float(np.nanmean(np.abs(delta) ** order))
                results.append((tau, float(order), structure_value))

        frame = pd.DataFrame(results, columns=["lag", "order", "structure_function"])
        return frame

    def increment_kurtosis(
        self,
        series: ArrayLike,
        *,
        lags: Sequence[int],
    ) -> pd.DataFrame:
        """Kurtosis (flatness) of magnetic increments as a function of lag."""
        data = pd.Series(series, dtype=float).dropna()
        if data.empty:
            raise ValueError("`series` must contain finite values.")

        results: List[Tuple[int, float]] = []
        values = data.to_numpy()
        n = len(values)

        for tau in lags:
            if tau <= 0 or tau >= n:
                raise ValueError("Each lag must satisfy 0 < tau < len(series).")
            delta = values[tau:] - values[:-tau]
            series_delta = pd.Series(delta)
            kurt = float(series_delta.kurtosis())
            results.append((tau, kurt))

        return pd.DataFrame(results, columns=["lag", "kurtosis"])

    def multi_spacecraft_pvi(
        self,
        spacecraft: Sequence[pd.DataFrame],
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Multi-point Partial Variance of Increments across spacecraft pairs.

        The denominator uses the root mean square magnetic field magnitude across
        all provided series.
        """
        if len(spacecraft) < 2:
            raise ValueError("At least two spacecraft are required.")

        target_columns = columns or self.config.magnetic_columns
        frames = [frame[list(target_columns)].astype(float) for frame in spacecraft]

        common_index = frames[0].index
        for frame in frames[1:]:
            common_index = common_index.intersection(frame.index)
        if common_index.empty:
            raise ValueError("Spacecraft dataframes do not share a common index.")

        frames = [frame.loc[common_index] for frame in frames]
        all_vectors = np.vstack([frame.to_numpy() for frame in frames])
        denominator = np.sqrt(np.mean(np.linalg.norm(all_vectors, axis=1) ** 2))
        if denominator == 0.0:
            raise ValueError("Root mean square magnetic magnitude is zero.")

        results = pd.DataFrame(index=common_index)
        labels = [f"C{idx+1}" for idx in range(len(frames))]

        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                diff = frames[i].to_numpy() - frames[j].to_numpy()
                magnitude = np.linalg.norm(diff, axis=1)
                column_name = f"PVI_{labels[i]}_{labels[j]}"
                results[column_name] = magnitude / denominator

        return results

    # ------------------------------------------------------------------ #
    # Minimum variance analysis and timing
    # ------------------------------------------------------------------ #

    def minimum_variance_analysis(
        self,
        df: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        demean: bool = True,
    ) -> MVAResult:
        """
        Perform minimum variance analysis on a magnetic-field interval.
        """
        target_columns = columns or self.config.magnetic_columns
        ensure_columns(df, target_columns)
        data = df[list(target_columns)].astype(float).to_numpy()
        if demean:
            data = data - data.mean(axis=0)

        covariance = np.cov(data, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        elongation = float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] != 0 else np.nan
        planarity = float(eigenvalues[2] / eigenvalues[1]) if eigenvalues[1] != 0 else np.nan

        return MVAResult(eigenvalues, eigenvectors, elongation, planarity)

    def timing_analysis(
        self,
        times: Sequence[pd.Timestamp | float | int],
        positions: Sequence[Sequence[float]],
    ) -> TimingResult:
        """
        Determine the normal vector and propagation velocity of a planar structure.

        Parameters
        ----------
        times:
            Iterable with four timestamps (or seconds) corresponding to the same
            event observed by each spacecraft.
        positions:
            Iterable with four position vectors (km or any consistent unit).
        """
        if len(times) != 4 or len(positions) != 4:
            raise ValueError("Exactly four times and four positions are required.")

        coords = np.asarray(positions, dtype=float)
        if coords.shape != (4, 3):
            raise ValueError("`positions` must have shape (4, 3).")

        if isinstance(times[0], pd.Timestamp) or isinstance(times[0], str):
            timestamps = pd.to_datetime(times)
            seconds = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps], dtype=float)
        else:
            seconds = np.asarray(times, dtype=float) - float(np.asarray(times, dtype=float)[0])

        offsets = coords - coords.mean(axis=0)
        time_offsets = seconds - seconds.mean()

        matrix = np.hstack((offsets, -time_offsets[:, np.newaxis]))
        _, _, vh = np.linalg.svd(matrix)
        solution = vh[-1]

        normal = solution[:3]
        velocity = solution[3]
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("Unable to determine a unique normal vector.")

        normal /= norm
        velocity /= norm
        if velocity < 0:
            normal = -normal
            velocity = -velocity

        residual = np.linalg.norm(matrix @ solution) / np.linalg.norm(matrix) if np.linalg.norm(matrix) else 0.0
        return TimingResult(normal, float(velocity), float(residual))

    # ------------------------------------------------------------------ #
    # Gradient-derived diagnostics
    # ------------------------------------------------------------------ #

    def _align_spacecraft(
        self,
        spacecraft: Sequence[pd.DataFrame],
    ) -> Tuple[pd.Index, List[pd.DataFrame]]:
        frames = [frame.copy() for frame in spacecraft]
        common_index = frames[0].index
        for frame in frames[1:]:
            common_index = common_index.intersection(frame.index)
        if common_index.empty:
            raise ValueError("Spacecraft data do not share a common index.")
        aligned = [frame.loc[common_index] for frame in frames]
        return common_index, aligned

    def _gradient_at_index(
        self,
        frames: Sequence[pd.DataFrame],
        idx: int,
        *,
        columns: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.vstack(
            [frame.iloc[idx][list(self.config.position_columns)].to_numpy(dtype=float) for frame in frames]
        )
        magnetics = np.vstack(
            [frame.iloc[idx][list(columns)].to_numpy(dtype=float) for frame in frames]
        )
        gradient, intercepts = linear_gradient(positions, magnetics)
        return positions, magnetics, gradient

    def calculate_divergence(
        self,
        spacecraft1: pd.DataFrame,
        spacecraft2: pd.DataFrame,
        spacecraft3: pd.DataFrame,
        spacecraft4: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.Series:
        """Estimate ``∇·B`` using four-point measurements."""
        frames = [spacecraft1, spacecraft2, spacecraft3, spacecraft4]
        target_columns = columns or self.config.magnetic_columns
        for frame in frames:
            ensure_columns(frame, (*self.config.position_columns, *target_columns))

        index, aligned = self._align_spacecraft(frames)
        divergences: List[float] = []

        for row in range(len(index)):
            _, _, gradient = self._gradient_at_index(aligned, row, columns=target_columns)
            divergence = float(np.trace(gradient))
            divergences.append(divergence)

        return pd.Series(divergences, index=index, name="divB")

    def magnetic_curvature_and_radius(
        self,
        spacecraft1: pd.DataFrame,
        spacecraft2: pd.DataFrame,
        spacecraft3: pd.DataFrame,
        spacecraft4: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate curvature of magnetic field lines from four-point measurements.
        """
        frames = [spacecraft1, spacecraft2, spacecraft3, spacecraft4]
        target_columns = columns or self.config.magnetic_columns
        for frame in frames:
            ensure_columns(frame, (*self.config.position_columns, *target_columns))

        index, aligned = self._align_spacecraft(frames)
        curvature_mag: List[float] = []
        radius: List[float] = []

        for row in range(len(index)):
            _, magnetics, gradient = self._gradient_at_index(aligned, row, columns=target_columns)
            b_mean = magnetics.mean(axis=0)
            b_norm = np.linalg.norm(b_mean)
            if b_norm == 0.0:
                curvature_mag.append(0.0)
                radius.append(np.inf)
                continue

            b_hat = b_mean / b_norm
            grad = gradient  # rows: components, cols: spatial derivatives
            db_hat = np.zeros_like(grad)

            for axis in range(3):
                dB = grad[:, axis]
                projection = np.dot(b_hat, dB)
                db_hat[:, axis] = (dB - b_hat * projection) / b_norm

            kappa_vec = db_hat @ b_hat
            kappa = float(np.linalg.norm(kappa_vec))
            curvature_mag.append(kappa)
            radius.append(np.inf if kappa == 0.0 else 1.0 / kappa)

        return pd.DataFrame(
            {
                "curvature": curvature_mag,
                "radius_of_curvature": radius,
            },
            index=index,
        )

    def current_helicity_components(
        self,
        spacecraft1: pd.DataFrame,
        spacecraft2: pd.DataFrame,
        spacecraft3: pd.DataFrame,
        spacecraft4: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute helicity density and the parallel/perpendicular components of ``J``.
        """
        frames = [spacecraft1, spacecraft2, spacecraft3, spacecraft4]
        target_columns = columns or self.config.magnetic_columns
        for frame in frames:
            ensure_columns(frame, (*self.config.position_columns, *target_columns))

        index, aligned = self._align_spacecraft(frames)
        helicity: List[float] = []
        j_parallel: List[float] = []
        j_perp: List[float] = []

        for row in range(len(index)):
            positions, magnetics, gradient = self._gradient_at_index(aligned, row, columns=target_columns)
            curl = np.array(
                [
                    gradient[2, 1] - gradient[1, 2],
                    gradient[0, 2] - gradient[2, 0],
                    gradient[1, 0] - gradient[0, 1],
                ]
            )
            current_density = curl / MU_0
            b_mean = magnetics.mean(axis=0)
            b_norm = np.linalg.norm(b_mean)
            if b_norm == 0.0:
                helicity.append(0.0)
                j_parallel.append(0.0)
                j_perp.append(float(np.linalg.norm(current_density)))
                continue

            b_hat = b_mean / b_norm
            j_par = float(np.dot(current_density, b_hat))
            j_parallel.append(j_par)
            j_perp_vector = current_density - j_par * b_hat
            j_perp.append(float(np.linalg.norm(j_perp_vector)))
            helicity.append(float(np.dot(b_mean, current_density)))

        return pd.DataFrame(
            {
                "helicity_density": helicity,
                "J_parallel": j_parallel,
                "J_perpendicular": j_perp,
            },
            index=index,
        )

    def tetrahedron_quality_metrics(
        self,
        spacecraft1: pd.DataFrame,
        spacecraft2: pd.DataFrame,
        spacecraft3: pd.DataFrame,
        spacecraft4: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate ``Q_G`` and ``Q_R`` quality factors for the spacecraft tetrahedron.
        """
        frames = [spacecraft1, spacecraft2, spacecraft3, spacecraft4]
        for frame in frames:
            ensure_columns(frame, self.config.position_columns)

        index, aligned = self._align_spacecraft(frames)
        qg_values: List[float] = []
        qr_values: List[float] = []
        volumes: List[float] = []

        for row in range(len(index)):
            positions = np.vstack(
                [frame.iloc[row][list(self.config.position_columns)].to_numpy(dtype=float) for frame in aligned]
            )
            q_g, q_r, volume = tetrahedron_quality_factors(positions)
            qg_values.append(q_g)
            qr_values.append(q_r)
            volumes.append(volume)

        return pd.DataFrame(
            {
                "Q_G": qg_values,
                "Q_R": qr_values,
                "tetrahedron_volume": volumes,
            },
            index=index,
        )

    # ------------------------------------------------------------------ #
    # Extreme value helpers
    # ------------------------------------------------------------------ #

    def stats_excess(self, data: ArrayLike, threshold: float) -> Tuple[float, float]:
        """Mean and standard deviation of excesses over a threshold."""
        excess = np.asarray(data, dtype=float)
        mask = excess > threshold
        if not np.any(mask):
            return 0.0, 0.0
        differences = excess[mask] - threshold
        return float(np.mean(differences)), float(np.std(differences))

    def plot_mean_excess(
        self,
        data: ArrayLike,
        min_threshold: float,
        max_threshold: float,
        *,
        num_thresholds: int = 100,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot the mean excess and error bars for a range of thresholds."""
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        mean_excesses, std_excesses = zip(
            *(self.stats_excess(data, threshold) for threshold in thresholds)
        )

        axis = ax or plt.gca()
        axis.errorbar(thresholds, mean_excesses, yerr=std_excesses, fmt="o", label="Mean Excess")
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Mean Excess")
        axis.legend(loc="best")
        axis.grid(True)
        if ax is None and self._matplotlib_backend_safe:
            plt.show()

    def fit_pot_model(
        self,
        data: ArrayLike,
        min_threshold: float,
        max_threshold: float,
        num_thresholds: int,
    ) -> pd.DataFrame:
        """Fit a Peaks Over Threshold model across several threshold values."""
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        series = np.asarray(data, dtype=float)

        results = []
        for threshold in thresholds:
            exceedances = series[series > threshold] - threshold
            if len(exceedances) == 0:
                continue
            shape, location, scale = genpareto.fit(exceedances)
            results.append(
                {
                    "Threshold": threshold,
                    "Shape": shape,
                    "Location": location,
                    "Scale": scale,
                }
            )

        return pd.DataFrame(results)

    def plot_shape_parameter(
        self,
        results: pd.DataFrame,
        *,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot the shape parameter returned by :meth:`fit_pot_model`."""
        axis = ax or plt.gca()
        axis.plot(results["Threshold"], results["Shape"], marker="o")
        axis.set_title("Evolution of Shape Parameter over Thresholds")
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Shape Parameter")
        axis.grid(True)
        if ax is None and self._matplotlib_backend_safe:
            plt.show()

    def plot_mean_residual_life(
        self,
        data: ArrayLike,
        thresholds: Iterable[float],
        *,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot the mean residual life to help select a POT threshold."""
        data = np.asarray(data, dtype=float)
        thresholds = np.asarray(list(thresholds), dtype=float)
        means = np.array([np.mean(data[data > threshold] - threshold) for threshold in thresholds])

        derivatives = np.gradient(means)
        change_rate = np.abs(np.gradient(derivatives))

        approx_start_idx = int(np.where(change_rate < 0.01)[0][0]) if np.any(change_rate < 0.01) else 0
        approx_threshold = thresholds[approx_start_idx]

        axis = ax or plt.gca()
        axis.plot(thresholds, means, marker="o")
        axis.axvline(x=approx_threshold, color="r", linestyle="--")
        axis.text(
            approx_threshold,
            float(means.max() / 2) if len(means) else 0.0,
            f"Linear begin ~{approx_threshold:.3f}",
            color="r",
        )
        axis.set_xlabel("Thresholds")
        axis.set_ylabel("Mean Excess")
        axis.set_title("Mean residual life plot")
        axis.grid(True)
        if ax is None and self._matplotlib_backend_safe:
            plt.show()

    def declustering_function(
        self,
        data: pd.DataFrame,
        *,
        column: str = "value",
        threshold: float = 30000,
        run: int = 10,
        plot: bool = False,
        ax: plt.Axes | None = None,
    ) -> pd.DataFrame:
        """
        Declustering routine for Peaks Over Threshold analysis.

        Parameters mirror the notebook implementation but expose a more explicit
        API.  The method returns a dataframe containing the maxima of each
        cluster above ``threshold``.
        """
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not present in dataframe.")

        working = data.copy()
        working["index"] = working.index
        pot_df = working[working[column] > threshold].copy()

        if pot_df.empty:
            return working.iloc[0:0]

        gaps = pot_df["index"].diff().fillna(run + 1)
        pot_df["cluster"] = (gaps > run).cumsum()

        decluster_idx = pot_df.groupby("cluster")[column].idxmax()
        declustered = working.loc[decluster_idx]

        if plot:
            axis = ax or plt.gca()
            below_mask = working[column] < threshold
            axis.scatter(working.index[below_mask], working.loc[below_mask, column], color="black")
            axis.scatter(
                working.index[~below_mask],
                working.loc[~below_mask, column],
                color="gray",
                alpha=0.5,
            )
            axis.scatter(declustered.index, declustered[column], color="red", label="Declustered POT")
            axis.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
            axis.set_xlabel("Index")
            axis.set_ylabel(column)
            axis.legend()
            if ax is None and self._matplotlib_backend_safe:
                plt.show()

        return declustered


class cosmic(CosmicAnalyzer):  # pragma: no cover - compatibility shim
    """Backward compatible alias for the previous lowercase class name."""

    # Older notebooks expected a module-level ``plt`` attribute directly on the
    # ``cosmic`` symbol; expose it here to keep monkeypatch-based tests working.
    plt = plt

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
