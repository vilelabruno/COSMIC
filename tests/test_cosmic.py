import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cosmic.cosmic as cosmic_module
from cosmic.cosmic import mu_0, cosmic


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
