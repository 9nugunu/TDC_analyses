from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader


def test_summarize_raw_csv_dataset(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "raw_250609_sweep_sparams_beforebrazing"
    folder.mkdir(parents=True)
    (folder / "in_0.5cell.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc1_S11;im:Trc1_S11;\n"
        "2600000000.0;1.0;0.0;\n2601000000.0;0.0;-1.0;\n",
        encoding="utf-8",
    )
    (folder / "out_1.5cell.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc1_S11;im:Trc1_S11;\n"
        "2600000000.0;1.0;0.0;\n",
        encoding="utf-8",
    )

    summary = DataLoader().summarize(folder)

    assert summary.to_dict("records") == [
        {
            "dataset_id": "raw_250609_sweep_sparams_beforebrazing",
            "data_kind": "experiment",
            "data_layer": "raw",
            "row_count": 3,
            "file_count": 2,
            "freq_min_ghz": 2.6,
            "freq_max_ghz": 2.601,
            "source_formats": ["raw_csv_ri"],
            "s_names": ["S11"],
            "tune_positions": [0.5, 1.5],
            "port_sides": ["in", "out"],
        }
    ]


def test_summarize_prepro_dataset_without_port_side(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "prepro_260415_sweep_sparams_fullbrazing"
    folder.mkdir(parents=True)
    (folder / "8_processed.csv").write_text(
        "freq[Hz],Magnitude,Phase_deg\n2756000000.0,-1.2,90\n",
        encoding="utf-8",
    )

    row = DataLoader().summarize(folder).iloc[0]

    assert row["dataset_id"] == "prepro_260415_sweep_sparams_fullbrazing"
    assert row["data_layer"] == "prepro"
    assert row["source_formats"] == ["processed_csv_db_phase"]
    assert row["tune_positions"] == [8.0]
    assert row["port_sides"] == []
