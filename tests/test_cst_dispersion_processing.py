from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.dispersion import (
    load_cst_dispersion_txt,
    process_cst_dispersion_txt,
    summarize_dispersion_modes,
    to_dispersion_wide_table,
)
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves
from deflector_tuning.visualization.plot_config import PlotConfig


def _write_cst_export(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#",
                '#"phase"\t"Mode 1 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.10",
                "90\t2.90",
                "120\t2.86",
                "180\t2.84",
                "#",
                '#"phase"\t"Mode 2 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.30",
                "90\t3.10",
                "120\t3.05",
                "180\t3.00",
            ]
        ),
        encoding="utf-8",
    )


def test_load_cst_dispersion_txt_parses_repeated_mode_sections(tmp_path: Path) -> None:
    source = tmp_path / "dispersion.txt"
    _write_cst_export(source)

    table = load_cst_dispersion_txt(source)

    assert list(table.columns) == ["source_file", "mode_index", "phase_deg", "freq_GHz"]
    assert len(table) == 8
    assert table["mode_index"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert table.loc[table["mode_index"] == 1, "freq_GHz"].tolist() == [3.10, 2.90, 2.86, 2.84]


def test_wide_and_summary_tables_keep_marker_frequency_contract(tmp_path: Path) -> None:
    source = tmp_path / "dispersion.txt"
    _write_cst_export(source)
    table = load_cst_dispersion_txt(source)

    wide = to_dispersion_wide_table(table)
    summary = summarize_dispersion_modes(table)

    assert list(wide.columns) == ["phase_deg", "mode_01_GHz", "mode_02_GHz"]
    first_mode = summary.loc[summary["mode_index"] == 1].iloc[0]
    assert first_mode["freq_90_GHz"] == pytest.approx(2.90)
    assert first_mode["freq_120_GHz"] == pytest.approx(2.86)
    assert first_mode["delta_120_minus_90_MHz"] == pytest.approx(-40.0)
    assert first_mode["monotonicity"] == "decreasing"


def test_process_cst_dispersion_txt_writes_standard_csvs(tmp_path: Path) -> None:
    source = tmp_path / "dispersion.txt"
    _write_cst_export(source)

    outputs = process_cst_dispersion_txt(source)

    assert outputs.long_csv.name == "dispersion_long.csv"
    assert outputs.wide_csv.exists()
    assert outputs.summary_csv.exists()
    summary = pd.read_csv(outputs.summary_csv)
    assert {"mode_index", "freq_90_GHz", "freq_120_GHz"}.issubset(summary.columns)


def test_process_cst_dispersion_txt_defaults_to_matching_prepro_dataset(tmp_path: Path) -> None:
    source = tmp_path / "data" / "sim" / "dispersion_case" / "dispersion.txt"
    source.parent.mkdir(parents=True)
    _write_cst_export(source)

    outputs = process_cst_dispersion_txt(source)

    assert outputs.long_csv.parent == tmp_path / "data" / "prepro" / "dispersion_case"
    assert outputs.summary_csv.exists()


def test_process_cst_dispersion_txt_keeps_explicit_output_dir(tmp_path: Path) -> None:
    source = tmp_path / "data" / "sim" / "dispersion_case" / "dispersion.txt"
    output_dir = tmp_path / "custom"
    source.parent.mkdir(parents=True)
    _write_cst_export(source)

    outputs = process_cst_dispersion_txt(source, output_dir=output_dir)

    assert outputs.long_csv.parent == output_dir


def test_plot_dispersion_curves_writes_png(tmp_path: Path) -> None:
    source = tmp_path / "dispersion.txt"
    _write_cst_export(source)
    table = load_cst_dispersion_txt(source)

    path = plot_dispersion_curves(table, tmp_path / "dispersion.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_dispersion_curves_uses_direct_labels_without_legend(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "dispersion.txt"
    _write_cst_export(source)
    table = load_cst_dispersion_txt(source)
    saved_figures = []

    def _capture_figure(fig, output_path, config=None):
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("figure", encoding="utf-8")
        return path

    monkeypatch.setattr("deflector_tuning.visualization.dispersion_plots.save_figure", _capture_figure)

    plot_dispersion_curves(table, tmp_path / "dispersion.png", config=PlotConfig(line_width=2.0, marker_size=64))

    ax = saved_figures[0].axes[0]
    assert ax.get_title() == "CST dispersion: frequency vs phase advance"
    assert ax.get_legend() is None
    assert min(line.get_linewidth() for line in ax.lines) >= 3.0
    assert {text.get_text() for text in ax.texts} == {"Mode 1", "Mode 2"}
