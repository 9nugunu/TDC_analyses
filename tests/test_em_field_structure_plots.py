from pathlib import Path

import pytest

from scripts.plot_em_field_phase_structure import default_output_path
import deflector_tuning.visualization.em_field_structure_plots as em_plots
from deflector_tuning.visualization.em_field_structure_plots import (
    _iris_center_guides,
    _phase_curve_guides,
    _profile_guide_color,
    _profile_guide_mode,
    _regular_cell_center_guides,
    _regular_group_guides,
    _structure_segments,
    load_field_phase_export,
    load_field_profile_export,
    plot_field_phase_with_tdc_structure,
    plot_field_profile_with_tdc_structure,
)
from deflector_tuning.visualization.plot_config import PlotConfig


def _field_phase_export(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#Parameters = {a=22.4409; b=58.192; d=29.148; t=5.84}",
                '#"Z / mm"\t"e-field (f=2.8565) (1)_Y (Z)_phase [Real]"',
                "#----------------------------------------------------",
                "-29.148\t-10.0",
                "0.0\t0.0",
                "29.148\t10.0",
                "#Parameters = {a=22.4409; b=58.192; d=29.148; t=5.84}",
                '#"Z / mm"\t"h-field (f=2.8565) (1)_X (Z)_phase [Real]"',
                "#----------------------------------------------------",
                "-29.148\t100.0",
                "0.0\t110.0",
                "29.148\t120.0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _field_profile_export(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#Parameters = {a=22.4409; b=58.192; d=29.148; t=5.84}",
                '#"Z / mm"\t"e-field (f=2.8565) (1)_Y (Z)_phase [Real]"',
                "#----------------------------------------------------",
                "-29.148\t-10.0",
                "0.0\t0.0",
                "29.148\t10.0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _h_field_profile_export(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#Parameters = {a=22.4409; b=58.192; d=29.148; t=5.84}",
                '#"Z / mm"\t"h-field (f=2.8565) (1)_X (Z) [Real]"',
                "#----------------------------------------------------",
                "-29.148\t0.0",
                "20.414\t1.0",
                "125.378\t0.5",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_field_phase_export_splits_e_and_h_sections(tmp_path: Path) -> None:
    export = load_field_phase_export(_field_phase_export(tmp_path / "field_phase.txt"))

    assert export.parameters["d"] == 29.148
    assert export.parameters["t"] == 5.84
    assert [trace.field_kind for trace in export.traces] == ["e", "h"]
    assert [trace.component for trace in export.traces] == ["Y", "X"]
    assert export.traces[0].label == "E field Y phase"
    assert export.traces[1].label == "H field X phase"
    assert export.traces[0].z_mm.tolist() == [-29.148, 0.0, 29.148]
    assert export.traces[1].phase_deg.tolist() == [100.0, 110.0, 120.0]


def test_plot_field_phase_with_tdc_structure_writes_png(tmp_path: Path) -> None:
    export = load_field_phase_export(_field_phase_export(tmp_path / "field_phase.txt"))

    path = plot_field_phase_with_tdc_structure(
        export,
        tmp_path / "field_phase_tdc_structure.png",
        config=PlotConfig(dpi=120),
    )

    assert path.exists()
    assert path.name == "field_phase_tdc_structure.png"


def test_plot_field_profile_uses_fill_band_when_structure_parameters_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = load_field_profile_export(_field_profile_export(tmp_path / "field_profile.txt"))
    band_calls = []
    guide_calls = []

    def record_band(*args, **kwargs):
        band_calls.append(kwargs)

    def record_guides(*args, **kwargs):
        guide_calls.append(kwargs)

    monkeypatch.setattr(em_plots, "_draw_tdc_half_section_band", record_band)
    monkeypatch.setattr(em_plots, "_draw_profile_guides", record_guides)

    path = plot_field_profile_with_tdc_structure(
        export,
        tmp_path / "field_profile_tdc_structure.png",
        config=PlotConfig(dpi=120),
    )

    assert path.exists()
    assert len(band_calls) == 1
    assert len(guide_calls) == 1
    assert band_calls[0]["axis_y"] < min(trace.values.min() for trace in export.traces)


def test_h_field_profile_uses_regular_cell_center_guides(tmp_path: Path) -> None:
    export = load_field_profile_export(_h_field_profile_export(tmp_path / "h_profile.txt"))

    assert _profile_guide_mode(export.traces) == "h_cell_centers"


def test_profile_guide_colors_follow_structure_color_families() -> None:
    assert _profile_guide_color("e_iris_centers") == "#c77855"
    assert _profile_guide_color("h_cell_centers") == "#4f8d63"


def test_structure_segments_anchor_input_coupler_end_at_zero() -> None:
    segments = _structure_segments(
        z_min=-58.296,
        z_max=379.028,
        cell_length=29.148,
        iris_thickness=5.84,
        iris_radius=22.4409,
        cell_radius=58.192,
        regular_cell_count=9,
    )

    labels = [segment[4] for segment in segments]
    by_label = {segment[4]: segment for segment in segments if segment[4]}

    assert by_label["Coupler IN"][:2] == pytest.approx((-29.148, 0.0))
    assert by_label["R1"][:2] == pytest.approx((5.84, 34.988))
    assert by_label["R9"][:2] == pytest.approx((285.744, 314.892))
    assert by_label["Coupler OUT"][:2] == pytest.approx((320.732, 349.88))
    assert labels.count("Coupler IN") == 1
    assert labels.count("Coupler OUT") == 1
    assert [label for label in labels if label.startswith("R")] == [f"R{index}" for index in range(1, 10)]
    assert labels.count("beam") == 2


def test_regular_group_guides_mark_every_three_regular_cells() -> None:
    guides = _regular_group_guides(cell_length=29.148, iris_thickness=5.84, regular_cell_count=9)

    assert [guide[0] for guide in guides] == pytest.approx([107.884, 212.848])
    assert [guide[1] for guide in guides] == ["R3", "R6"]


def test_regular_cell_center_guides_mark_every_regular_cell_center() -> None:
    guides = _regular_cell_center_guides(cell_length=29.148, iris_thickness=5.84, regular_cell_count=9)

    assert [guide[0] for guide in guides] == pytest.approx(
        [20.414, 55.402, 90.39, 125.378, 160.366, 195.354, 230.342, 265.33, 300.318]
    )
    assert [guide[1] for guide in guides] == [f"R{index}" for index in range(1, 10)]


def test_iris_center_guides_mark_every_iris_center() -> None:
    guides = _iris_center_guides(cell_length=29.148, iris_thickness=5.84, regular_cell_count=9)

    assert [guide[0] for guide in guides] == pytest.approx(
        [2.92, 37.908, 72.896, 107.884, 142.872, 177.86, 212.848, 247.836, 282.824, 317.812]
    )
    assert [guide[1] for guide in guides] == ["IN iris", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "OUT iris"]


def test_phase_curve_guides_include_input_and_output_iris_boundaries() -> None:
    guides = _phase_curve_guides(cell_length=29.148, iris_thickness=5.84, regular_cell_count=9)

    assert [guide[0] for guide in guides] == pytest.approx([2.92, 107.884, 212.848, 317.812])
    assert [guide[1] for guide in guides] == ["IN iris", "R3", "R6", "OUT iris"]


def test_default_output_path_uses_fig_analyses_tree() -> None:
    path = default_output_path(
        Path("data") / "sim" / "sim_EMfield_260618_PhaseDistribution" / "EM_fieldPhase.txt"
    )

    assert path == (
        Path("fig")
        / "analyses"
        / "sim_EMfield_260618_PhaseDistribution"
        / "figures"
        / "em_field_phase"
        / "field_phase_tdc_structure.png"
    )
