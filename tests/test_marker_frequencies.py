from pathlib import Path

import pytest

from deflector_tuning.markers.frequency_markers import extract_marker_frequencies
def _write_dispersion_summary(folder: Path) -> Path:
    processed = folder / "processed"
    processed.mkdir(parents=True)
    summary = processed / "dispersion_summary.csv"
    summary.write_text(
        "mode_index,freq_90_GHz,freq_120_GHz\n"
        "1,2.8778045932946,2.8574021866048\n"
        "2,2.9008737821024,2.8805631272866\n",
        encoding="utf-8",
    )
    return summary
def test_extract_sim_marker_frequencies_from_mode_1_dispersion_summary(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_260505_dispersion_single_cell_step1"
    _write_dispersion_summary(folder)

    markers = extract_marker_frequencies(folder, marker_role="sim")

    assert markers.to_dict("records") == [
        {
            "marker_name": "f_2pi3",
            "freq_ghz": 2.8574021866048,
            "marker_role": "sim",
            "marker_source": "dispersion_mode_1",
            "source_file": "processed/dispersion_summary.csv",
            "uncorrected_freq_ghz": 2.8574021866048,
            "frequency_scale_factor": 1.0,
            "temp_op_C": None,
            "temp_meas_C": None,
            "humidity_fraction": None,
            "thermal_alpha_per_C": None,
            "eps_air_humid": None,
        },
        {
            "marker_name": "f_mean",
            "freq_ghz": pytest.approx(2.8676033899497),
            "marker_role": "sim",
            "marker_source": "dispersion_mode_1",
            "source_file": "processed/dispersion_summary.csv",
            "uncorrected_freq_ghz": pytest.approx(2.8676033899497),
            "frequency_scale_factor": 1.0,
            "temp_op_C": None,
            "temp_meas_C": None,
            "humidity_fraction": None,
            "thermal_alpha_per_C": None,
            "eps_air_humid": None,
        },
        {
            "marker_name": "f_pi2",
            "freq_ghz": 2.8778045932946,
            "marker_role": "sim",
            "marker_source": "dispersion_mode_1",
            "source_file": "processed/dispersion_summary.csv",
            "uncorrected_freq_ghz": 2.8778045932946,
            "frequency_scale_factor": 1.0,
            "temp_op_C": None,
            "temp_meas_C": None,
            "humidity_fraction": None,
            "thermal_alpha_per_C": None,
            "eps_air_humid": None,
        },
    ]
def test_extract_exp_marker_frequencies_apply_temperature_humidity_correction(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_260505_dispersion_single_cell_step1"
    _write_dispersion_summary(folder)

    markers = extract_marker_frequencies(folder, marker_role="exp")

    scale = 0.999569925074132
    by_name = markers.set_index("marker_name")
    assert by_name.loc["f_2pi3", "freq_ghz"] == pytest.approx(2.8574021866048 * scale)
    assert by_name.loc["f_mean", "freq_ghz"] == pytest.approx(2.8676033899497 * scale)
    assert by_name.loc["f_pi2", "freq_ghz"] == pytest.approx(2.8778045932946 * scale)
    assert by_name.loc["f_2pi3", "marker_role"] == "exp"
    assert by_name.loc["f_2pi3", "marker_source"] == "dispersion_mode_1_temp_humidity_corrected"
    assert by_name.loc["f_2pi3", "frequency_scale_factor"] == pytest.approx(scale)
    assert by_name.loc["f_2pi3", "temp_op_C"] == 20.0
    assert by_name.loc["f_2pi3", "temp_meas_C"] == 24.4
    assert by_name.loc["f_2pi3", "humidity_fraction"] == 0.65
