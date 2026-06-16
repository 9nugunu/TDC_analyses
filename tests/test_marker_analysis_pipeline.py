from pathlib import Path

import pandas as pd

import deflector_tuning.analysis.marker_pipeline as marker_pipeline
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
    for tune_position, phases in [(0.5, [10.0, 20.0, 30.0]), (1.5, [-110.0, -100.0, -90.0])]:
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
    assert set(result["marker_points"]["source_file"]) == {"0.5_processed.csv", "1.5_processed.csv"}
    assert "s_phase_deg" in result["marker_points"].columns
    first_phase = result["phase_advance"].sort_values("marker_name").iloc[0]
    assert first_phase["from_tune_position"] == 0.5
    assert first_phase["to_tune_position"] == 1.5
    assert first_phase["position_family"] == "cell"


def test_save_marker_analysis_writes_csv_tables(tmp_path: Path) -> None:
    result = {
        "markers": pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
        "marker_points": pd.DataFrame(
            [{"marker_name": "f_2pi3", "s_phase_deg": 10.0, "data_kind": "experiment", "port_side": pd.NA}]
        ),
        "phase_advance": pd.DataFrame(
            [{"marker_name": "f_2pi3", "phase_error_from_240_deg": 0.0, "data_kind": "experiment", "port_side": pd.NA}]
        ),
        "phase_summary": pd.DataFrame(
            [{"marker_name": "f_2pi3", "transition_count": 1, "data_kind": "experiment", "port_side": pd.NA}]
        ),
    }
    output_dir = tmp_path / "analysis_outputs"

    paths = save_marker_analysis(result, output_dir)

    assert list(paths.keys()) == ["markers", "marker_points", "phase_advance", "phase_summary"]
    for path in paths.values():
        assert path.exists()
    saved_marker_points = pd.read_csv(paths["marker_points"])
    assert saved_marker_points.loc[0, "s_phase_deg"] == 10.0
    for table_name in ["marker_points", "phase_advance", "phase_summary"]:
        saved = pd.read_csv(paths[table_name])
        assert "data_kind" not in saved.columns
        assert "port_side" not in saved.columns


def test_build_marker_analysis_samples_only_s11_rows(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, pd.DataFrame] = {}

    class FakeLoader:
        def load(self, path):
            return pd.DataFrame(
                [
                    {"source_file": "0.5.s2p", "s_name": "S11", "freq_ghz": 2.856},
                    {"source_file": "0.5.s2p", "s_name": "S21", "freq_ghz": 2.856},
                ]
            )

    def fake_sample_nearest_markers(sparameter_table: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
        captured["sparameter_table"] = sparameter_table
        return pd.DataFrame(
            [
                {
                    "dataset_id": "dataset",
                    "data_kind": "experiment",
                    "data_layer": "prepro",
                    "source_file": "0.5.s2p",
                    "tune_position": 0.5,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "exp",
                    "target_freq_ghz": 2.856,
                    "freq_ghz": 2.856,
                    "s_db": -1.0,
                    "s_phase_deg": 10.0,
                },
                {
                    "dataset_id": "dataset",
                    "data_kind": "experiment",
                    "data_layer": "prepro",
                    "source_file": "1.5.s2p",
                    "tune_position": 1.5,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "exp",
                    "target_freq_ghz": 2.856,
                    "freq_ghz": 2.856,
                    "s_db": -2.0,
                    "s_phase_deg": -110.0,
                },
            ]
        )

    monkeypatch.setattr(
        marker_pipeline,
        "extract_marker_frequencies",
        lambda *args, **kwargs: pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
    )
    monkeypatch.setattr(marker_pipeline, "sample_nearest_markers", fake_sample_nearest_markers)

    build_marker_analysis(
        sparameter_path=tmp_path / "data" / "prepro" / "dataset",
        dispersion_path=tmp_path / "data" / "sim" / "dispersion",
        marker_role="exp",
        loader=FakeLoader(),
    )

    assert captured["sparameter_table"]["s_name"].tolist() == ["S11"]
