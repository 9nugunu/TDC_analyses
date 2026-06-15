from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.marker_pipeline import build_marker_analysis, save_marker_analysis


def _write_dispersion_summary(folder: Path) -> None:
    processed = folder / "processed"
    processed.mkdir(parents=True)
    (processed / "dispersion_summary.csv").write_text(
        "mode_index,freq_90_GHz,freq_120_GHz\n"
        "1,2.8778045932946,2.8574021866048\n",
        encoding="utf-8",
    )


def _write_prepro_dataset(folder: Path) -> None:
    folder.mkdir(parents=True)
    for tune_position, phases in [(0.5, [10.0, 20.0, 30.0]), (1.0, [-110.0, -100.0, -90.0])]:
        filename = folder / f"{tune_position}_processed.csv"
        filename.write_text(
            "freq[Hz],Magnitude,Phase_deg\n"
            f"2855880000,-1.0,{phases[0]}\n"
            f"2866050000,-2.0,{phases[1]}\n"
            f"2876210000,-3.0,{phases[2]}\n",
            encoding="utf-8",
        )


def test_build_marker_analysis_processes_one_folder_into_marker_phase_tables(tmp_path: Path) -> None:
    sparameter_path = tmp_path / "data" / "prepro" / "sample_prepro"
    dispersion_path = tmp_path / "data" / "sim" / "260505_single_cell_dispersion_step1"
    _write_prepro_dataset(sparameter_path)
    _write_dispersion_summary(dispersion_path)

    result = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role="exp",
    )

    assert list(result.keys()) == ["markers", "marker_points", "phase_advance", "phase_summary"]
    assert len(result["markers"]) == 3
    assert len(result["marker_points"]) == 6
    assert len(result["phase_advance"]) == 3
    assert len(result["phase_summary"]) == 3
    assert set(result["marker_points"]["source_file"]) == {"0.5_processed.csv", "1.0_processed.csv"}
    assert "s_phase_deg" in result["marker_points"].columns
    first_phase = result["phase_advance"].sort_values("marker_name").iloc[0]
    assert first_phase["from_tune_position"] == 0.5
    assert first_phase["to_tune_position"] == 1.0


def test_save_marker_analysis_writes_csv_tables(tmp_path: Path) -> None:
    result = {
        "markers": pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
        "marker_points": pd.DataFrame([{"marker_name": "f_2pi3", "s_phase_deg": 10.0}]),
        "phase_advance": pd.DataFrame([{"marker_name": "f_2pi3", "phase_error_from_240_deg": 0.0}]),
        "phase_summary": pd.DataFrame([{"marker_name": "f_2pi3", "transition_count": 1}]),
    }
    output_dir = tmp_path / "analysis_outputs"

    paths = save_marker_analysis(result, output_dir)

    assert list(paths.keys()) == ["markers", "marker_points", "phase_advance", "phase_summary"]
    for path in paths.values():
        assert path.exists()
    assert pd.read_csv(paths["marker_points"]).loc[0, "s_phase_deg"] == 10.0
