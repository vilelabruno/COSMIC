"""
Core analysis utilities for the COSMIC project.

This module exposes :class:`CosmicAnalyzer`, a battery of routines used to
process magnetometer data from the ESA Cluster mission.  The original project
started life inside notebooks; the code below reshapes those experiments into a
clean, reusable, and well documented Python API ready for production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d
from scipy.stats import genpareto

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
        window: int = 50,
    ) -> pd.Series:
        """
        Rolling standard deviation of log-returns for the supplied column.
        """
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
