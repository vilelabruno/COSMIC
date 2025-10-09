import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cosmic.cosmic as cosmic_module
from cosmic.cosmic import MVAResult, SpectrumResult, TimingResult, mu_0, cosmic
from cosmic.utils import detect_outliers, median_absolute_deviation


POS_COLS = [
    "sc_pos_xyz_gse__C1_CP_FGM_FULL1",
    "sc_pos_xyz_gse__C1_CP_FGM_FULL2",
    "sc_pos_xyz_gse__C1_CP_FGM_FULL3",
]

B_COLS = [
    "B_vec_xyz_gse__C1_CP_FGM_FULL1",
    "B_vec_xyz_gse__C1_CP_FGM_FULL2",
    "B_vec_xyz_gse__C1_CP_FGM_FULL3",
]


@pytest.fixture(autouse=True)
def disable_matplotlib_show(monkeypatch):
    """Avoid opening GUI backends during the tests."""
    monkeypatch.setattr(cosmic_module.plt, "show", lambda: None)


@pytest.fixture
def cosmic_obj():
    return cosmic(matplotlib_backend_safe=False)


def make_spacecraft_df(position, magnetic_field):
    data = {POS_COLS[i]: [position[i]] for i in range(3)}
    data.update({B_COLS[i]: [magnetic_field[i]] for i in range(3)})
    return pd.DataFrame(data)


@pytest.fixture
def sample_dat_file(tmp_path: Path):
    content = "\n".join(
        [
            "START_VARIABLE = sc_pos_xyz_gse__C1_CP_FGM_FULL",
            "SIZES = 3",
            "START_VARIABLE = B_vec_xyz_gse__C1_CP_FGM_FULL",
            "SIZES = 3",
            "START_VARIABLE = extra_var",
            "SIZES = 5",
            "0,0,0,1,0,0,9,8,7,6,5",
            "1,1,1,0,1,0,4,3,2,1,0",
            "EOF",
        ]
    )
    path = tmp_path / "sample.dat"
    path.write_text(content)
    return path


def test_ler_arquivo_dat_parses_dat_file(cosmic_obj, sample_dat_file):
    df = cosmic_obj.ler_arquivo_dat(str(sample_dat_file))

    expected_columns = (
        [f"sc_pos_xyz_gse__C1_CP_FGM_FULL{i}" for i in range(1, 4)]
        + [f"B_vec_xyz_gse__C1_CP_FGM_FULL{i}" for i in range(1, 4)]
        + [f"extra_var{i}" for i in range(1, 6)]
    )

    assert list(df.columns) == expected_columns
    assert df.shape == (2, 11)
    np.testing.assert_array_equal(
        df[POS_COLS].astype(float).to_numpy(),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def test_calculate_differences(cosmic_obj):
    df1 = make_spacecraft_df([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    df2 = make_spacecraft_df([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])

    B_diff = cosmic_obj.calculate_B_diff(df1, df2)
    r_diff = cosmic_obj.calculate_r_diff(df1, df2)

    np.testing.assert_array_almost_equal(B_diff, np.array([[1.0, -1.0, 0.0]]))
    np.testing.assert_array_almost_equal(r_diff, np.array([[-1.0, -2.0, -3.0]]))


def test_calculate_current_density_matches_expected_value(cosmic_obj):
    df1 = make_spacecraft_df([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    df2 = make_spacecraft_df([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    df3 = make_spacecraft_df([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

    result = cosmic_obj.calculate_current_density(df1, df2, df3)

    expected_value = 2.0 / mu_0
    assert pytest.approx(expected_value) == result.iloc[0]


def test_curlometer_combines_current_densities(cosmic_obj):
    df1 = make_spacecraft_df([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    df2 = make_spacecraft_df([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    df3 = make_spacecraft_df([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    df4 = make_spacecraft_df([0.0, 0.0, 1.0], [1.0, 1.0, 1.0])

    curl = cosmic_obj.curlometer(df1, df2, df3, df4)

    expected_value = 2.0 / (3.0 * mu_0)
    assert pytest.approx(expected_value) == curl.iloc[0]


def test_calculate_mod_B_returns_magnitude(cosmic_obj):
    data = pd.DataFrame(
        {
            B_COLS[0]: [3.0, 0.0],
            B_COLS[1]: [4.0, 0.0],
            B_COLS[2]: [0.0, 5.0],
        }
    )

    result = cosmic_obj.calculate_mod_B(
        data, B_COLS[0], B_COLS[1], B_COLS[2]
    )

    np.testing.assert_allclose(result.to_numpy(), np.array([5.0, 5.0]))


def test_calculate_PVI_normalises_differences(cosmic_obj):
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    pvi = cosmic_obj.calculate_PVI(series, tau=2)

    assert len(pvi) == 4
    np.testing.assert_allclose(pvi.to_numpy(), np.ones(4))


@pytest.mark.parametrize(
    "vectors, expected",
    [
        (
            np.vstack(
                (
                    np.repeat([[1.0, 0.0, 0.0]], 3, axis=0),
                    np.repeat([[-1.0, 0.0, 0.0]], 3, axis=0),
                )
            ),
            1,
        ),
        (
            np.repeat([[1.0, 0.0, 0.0]], 6, axis=0),
            0,
        ),
    ],
)
def test_cs_detection_identifies_current_sheets(cosmic_obj, vectors, expected):
    df = pd.DataFrame(vectors, columns=["Bx", "By", "Bz"])
    result = cosmic_obj.cs_detection(df, tau=3, theta_c=90.0)
    assert result == expected


def test_limethod_returns_detection_series(cosmic_obj):
    first_segment = np.repeat([[1.0, 0.0, 0.0]], 11, axis=0)
    second_segment = np.repeat([[-1.0, 0.0, 0.0]], 11, axis=0)
    tail_segment = np.repeat([[-1.0, 0.0, 0.0]], 5, axis=0)
    data = np.vstack((first_segment, second_segment, tail_segment))

    index = pd.date_range("2000-01-01", periods=len(data), freq="S")
    df = pd.DataFrame(data, index=index, columns=["Bx", "By", "Bz"])

    detections = cosmic_obj.limethod(df, theta_c=90.0, tau_sec=0.5)

    assert len(detections) == len(df) - 2 * int(22 * 0.5)
    assert detections.columns.tolist() == ["Time", "cs_out"]
    assert detections["cs_out"].unique().tolist() == [1]


def test_convert_to_float_handles_d_notation(cosmic_obj):
    assert cosmic_obj.convert_to_float("1.23D+02") == pytest.approx(123.0)
    assert cosmic_obj.convert_to_float("4.56E-01") == pytest.approx(0.456)


def test_calculate_magnetic_volatility_uses_log_returns(cosmic_obj):
    df = pd.DataFrame({"mag": [1.0, 2.0, 4.0, 8.0, 16.0]})

    volat = cosmic_obj.calculate_magnetic_volatility(df, "mag", tau=1, w=2)

    assert math.isnan(volat.iloc[0])
    assert math.isnan(volat.iloc[1])
    np.testing.assert_allclose(volat.iloc[2:].to_numpy(), np.zeros(3))


def test_apply_gaussian_kernel_preserves_constant_signal(cosmic_obj):
    constant_signal = np.ones(5)
    smoothed = cosmic_obj.apply_gaussian_kernel(constant_signal, sigma=1.0)
    np.testing.assert_allclose(smoothed, constant_signal)


def test_fit_pot_model_returns_dataframe(cosmic_obj):
    data = pd.Series(np.linspace(1.0, 10.0, 100))

    results = cosmic_obj.fit_pot_model(data, min_threshold=5.0, max_threshold=9.0, num_thresholds=4)

    assert list(results.columns) == ["Threshold", "Shape", "Location", "Scale"]
    assert len(results) == 4
    np.testing.assert_allclose(results["Threshold"].to_numpy(), np.linspace(5.0, 9.0, 4))


def test_plotting_functions_execute_without_errors(cosmic_obj):
    yy = pd.Series([1.0, 2.0, 3.0])
    cosmic_obj.plot_mod_B(yy)
    cosmic_obj.plot_data(yy)

    data = np.linspace(10.0, 20.0, 50)
    cosmic_obj.plot_mean_excess(data, min_threshold=9.0, max_threshold=10.0, num_thresholds=5)

    results = pd.DataFrame(
        {
            "Threshold": [0.1, 0.2, 0.3],
            "Shape": [0.0, 0.1, 0.2],
            "Location": [0.0, 0.0, 0.0],
            "Scale": [1.0, 1.0, 1.0],
        }
    )
    cosmic_obj.plot_shape_parameter(results)

    thresholds = np.linspace(1.0, 2.0, 5)
    cosmic_obj.plot_mean_residual_life(np.full(100, 5.0), thresholds)


def test_declustering_function_returns_cluster_maxima(cosmic_obj):
    data = pd.DataFrame(
        {
            "value": [
                10000,
                32000,
                31000,
                12000,
                13000,
                33000,
                34000,
            ]
        }
    )

    declustered = cosmic_obj.declustering_function(data, threshold=30000, run=2)

    assert declustered["value"].tolist() == [32000, 34000]
    assert declustered.index.tolist() == [1, 6]


def test_calculate_magnetic_energy_density_returns_expected_values(cosmic_obj):
    df = pd.DataFrame(
        {
            B_COLS[0]: [1.0, 0.0],
            B_COLS[1]: [0.0, 2.0],
            B_COLS[2]: [0.0, 0.0],
        }
    )
    energy = cosmic_obj.calculate_magnetic_energy_density(df)
    expected = np.array([1.0, 4.0]) / (2.0 * mu_0)
    np.testing.assert_allclose(energy.to_numpy(), expected)


def test_normalize_magnetic_field_standardises_columns(cosmic_obj):
    df = pd.DataFrame(
        {
            B_COLS[0]: [1.0, 2.0, 3.0],
            B_COLS[1]: [2.0, 4.0, 6.0],
            B_COLS[2]: [3.0, 6.0, 9.0],
            "other": [10, 20, 30],
        }
    )
    normalised = cosmic_obj.normalize_magnetic_field(df)
    for column in B_COLS:
        series = normalised[column]
        assert pytest.approx(0.0) == float(series.mean())
        assert pytest.approx(1.0) == float(series.std(ddof=0))
    assert normalised["other"].tolist() == df["other"].tolist()


def test_remove_outliers_interpolates_with_time_index(cosmic_obj):
    index = pd.date_range("2020-01-01", periods=5, freq="S")
    df = pd.DataFrame(
        {
            B_COLS[0]: [1.0, 2.0, 100.0, 2.0, 1.0],
            B_COLS[1]: [0.0, 0.0, 0.0, 0.0, 0.0],
            B_COLS[2]: [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    cleaned = cosmic_obj.remove_outliers(df, columns=[B_COLS[0]], threshold=3.0)
    assert pytest.approx(cleaned.iloc[2][B_COLS[0]]) == 2.0


def test_remove_outliers_allows_median_fill(cosmic_obj):
    df = pd.DataFrame(
        {
            B_COLS[0]: [1.0, 2.0, 40.0, 2.0, 1.0],
            B_COLS[1]: [0.0, 0.0, 0.0, 0.0, 0.0],
            B_COLS[2]: [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    cleaned = cosmic_obj.remove_outliers(
        df,
        columns=[B_COLS[0]],
        threshold=3.0,
        fill_strategy="median",
    )
    median_value = df[B_COLS[0]].median()
    assert pytest.approx(cleaned.iloc[2][B_COLS[0]]) == median_value


def test_resample_dataframe_changes_frequency(cosmic_obj):
    index = pd.date_range("2020-01-01", periods=6, freq="S")
    df = pd.DataFrame(
        {
            B_COLS[0]: np.arange(6, dtype=float),
            B_COLS[1]: np.arange(6, dtype=float),
            B_COLS[2]: np.arange(6, dtype=float),
        },
        index=index,
    )
    resampled = cosmic_obj.resample_dataframe(df, "2S", agg="mean")
    assert len(resampled) == 3
    np.testing.assert_allclose(resampled[B_COLS[0]].to_numpy(), np.array([0.5, 2.5, 4.5]))


def test_mad_helpers_flag_expected_outliers():
    series = pd.Series([1.0, 1.2, 0.9, 15.0])
    mad = median_absolute_deviation(series)
    assert mad > 0
    mask = detect_outliers(series, threshold=3.0)
    assert mask.tolist() == [False, False, False, True]


def test_power_spectral_density_returns_slope(cosmic_obj):
    sample_frequency = 10.0
    time = np.arange(0, 100, 1 / sample_frequency)
    series = np.sin(2 * np.pi * 1.0 * time) + 0.1 * np.random.default_rng(0).normal(size=time.size)

    result = cosmic_obj.power_spectral_density(
        series,
        sample_frequency,
        slope_range=(0.5, 2.0),
    )

    assert isinstance(result, SpectrumResult)
    assert result.slope is not None
    assert result.slope_range == (0.5, 2.0)


def test_component_power_spectra_handles_parallel_perpendicular(cosmic_obj):
    sample_frequency = 5.0
    time = np.arange(0, 20, 1 / sample_frequency)
    bx = 5.0 + np.sin(2 * np.pi * 0.2 * time)
    by = 1.0 + 0.1 * np.cos(2 * np.pi * 0.5 * time)
    bz = 0.5 + 0.05 * np.sin(2 * np.pi * 0.8 * time)
    df = pd.DataFrame({B_COLS[0]: bx, B_COLS[1]: by, B_COLS[2]: bz})

    spectra = cosmic_obj.component_power_spectra(df, sample_frequency)

    for key in ("total", "parallel", "perpendicular"):
        assert key in spectra
        spec = spectra[key]
        assert isinstance(spec, SpectrumResult)
        assert len(spec.frequencies_hz) == len(spec.psd)


def test_autocorrelation_identifies_decorrelation_lag(cosmic_obj):
    series = [1.0, 0.0, 0.0, 0.0]
    lags, corr, decorrelation = cosmic_obj.autocorrelation(series)

    assert lags[0] == 0
    assert corr[0] == pytest.approx(1.0)
    assert decorrelation == 1


def test_cross_correlation_recovers_time_shift(cosmic_obj):
    fs = 20.0
    time = np.arange(0, 1.0, 1 / fs)
    base = np.sin(2 * np.pi * 3.0 * time)
    shifted = np.roll(base, 4)

    lags, corr, best_lag_seconds = cosmic_obj.cross_correlation(base, shifted, fs)

    assert np.argmax(np.abs(corr)) == list(lags).index(-4)
    assert pytest.approx(best_lag_seconds, rel=1e-3) == -4 / fs


def test_structure_functions_matches_linear_series(cosmic_obj):
    series = np.arange(10, dtype=float)
    result = cosmic_obj.structure_functions(series, orders=(2,), lags=(1,))
    assert result.shape == (1, 3)
    assert pytest.approx(result.iloc[0]["structure_function"]) == 1.0


def test_increment_kurtosis_handles_multiple_lags(cosmic_obj):
    series = np.sin(np.linspace(0, 2 * np.pi, 50))
    result = cosmic_obj.increment_kurtosis(series, lags=(1, 2, 3))
    assert result["lag"].tolist() == [1, 2, 3]
    assert result["kurtosis"].notna().all()


def create_spacecraft_frame(position, magnetic):
    data = {POS_COLS[i]: position[i] for i in range(3)}
    data.update({B_COLS[i]: magnetic[i] for i in range(3)})
    return pd.DataFrame(data, index=pd.Index([0]))


def test_multi_spacecraft_pvi_returns_zero_for_identical_fields(cosmic_obj):
    frames = [
        create_spacecraft_frame([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
        create_spacecraft_frame([1.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
    ]
    pvi = cosmic_obj.multi_spacecraft_pvi(frames)
    assert pvi.shape == (1, 1)
    assert pytest.approx(pvi.iloc[0, 0]) == 0.0


def test_minimum_variance_analysis_recovers_principal_axes(cosmic_obj):
    data = np.eye(3)
    df = pd.DataFrame(data, columns=B_COLS)
    result = cosmic_obj.minimum_variance_analysis(df)
    assert isinstance(result, MVAResult)
    assert np.all(np.diff(result.eigenvalues) <= 0)


def test_timing_analysis_recovers_normal_and_velocity(cosmic_obj):
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    times = [0.0, 0.1, 0.0, 0.0]
    result = cosmic_obj.timing_analysis(times, positions)
    assert isinstance(result, TimingResult)
    assert pytest.approx(result.velocity, rel=1e-5) == 10.0
    assert np.allclose(result.normal, np.array([1.0, 0.0, 0.0]), atol=1e-5)


def make_linear_spacecraft_frames():
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    fields = positions
    frames = [
        create_spacecraft_frame(pos, field) for pos, field in zip(positions, fields)
    ]
    return frames


def test_calculate_divergence_matches_expected_value(cosmic_obj):
    frames = make_linear_spacecraft_frames()
    divergence = cosmic_obj.calculate_divergence(*frames)
    assert divergence.iloc[0] == pytest.approx(3.0, rel=1e-5)


def test_magnetic_curvature_zero_for_uniform_field(cosmic_obj):
    frames = [
        create_spacecraft_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        create_spacecraft_frame([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        create_spacecraft_frame([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        create_spacecraft_frame([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ]
    curvature = cosmic_obj.magnetic_curvature_and_radius(*frames)
    assert pytest.approx(curvature.iloc[0]["curvature"]) == 0.0
    assert np.isinf(curvature.iloc[0]["radius_of_curvature"])


def test_current_helicity_zero_for_uniform_field(cosmic_obj):
    frames = [
        create_spacecraft_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        create_spacecraft_frame([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        create_spacecraft_frame([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        create_spacecraft_frame([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ]
    helicity = cosmic_obj.current_helicity_components(*frames)
    assert helicity.iloc[0]["helicity_density"] == pytest.approx(0.0)
    assert helicity.iloc[0]["J_parallel"] == pytest.approx(0.0)
    assert helicity.iloc[0]["J_perpendicular"] == pytest.approx(0.0)


def test_tetrahedron_quality_metrics_regular_configuration(cosmic_obj):
    positions = [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ]
    frames = [
        create_spacecraft_frame(pos, [1.0, 0.0, 0.0]) for pos in positions
    ]
    quality = cosmic_obj.tetrahedron_quality_metrics(*frames)
    assert quality.iloc[0]["Q_G"] == pytest.approx(3.0, rel=1e-3)
    assert quality.iloc[0]["Q_R"] == pytest.approx(1.0, rel=1e-3)
