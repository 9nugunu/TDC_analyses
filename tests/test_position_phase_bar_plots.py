from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from deflector_tuning.visualization.position_phase_bar_plots import (
    ADVANCE_IDEAL_BAR_WIDTH,
    ADVANCE_MEASURED_BAR_WIDTH,
    ADVANCE_IDEAL_COLOR,
    POSITION_IDEAL_COLOR,
    build_position_phase_bar_table,
    build_position_phase_target_table,
    classify_phase_advance_difference,
    plot_position_phase_advance_bars,
    plot_position_phase_bars,
)
from deflector_tuning.visualization.plot_config import PlotConfig


def _marker_points() -> pd.DataFrame:
    phases = {
        1.0: {
            "f_2pi3": -165.2797634916735,
            "f_mean": -178.74461937073156,
            "f_pi2": 169.48012254940267,
        },
        2.0: {
            "f_2pi3": 74.40030955943195,
            "f_mean": 26.81148233359222,
            "f_pi2": -35.29997402642941,
        },
    }
    frequencies = {
        "f_2pi3": 2.8558872595868734,
        "f_mean": 2.8660473089425436,
        "f_pi2": 2.8762073582982137,
    }
    return pd.DataFrame(
        [
            {
                "dataset_id": "raw_sweep_sample",
                "source_file": f"{position:g}_portE.S2P",
                "tune_position": position,
                "marker_name": marker_name,
                "freq_target_ghz": frequencies[marker_name],
                "freq_ghz": frequencies[marker_name] - 0.000007,
                "s_phase_deg": phase,
            }
            for position, marker_phases in phases.items()
            for marker_name, phase in marker_phases.items()
        ]
    )


def test_build_position_phase_bar_table_preserves_signed_phase_and_order() -> None:
    result = build_position_phase_bar_table(_marker_points(), positions=(1.0, 2.0))

    assert list(result.columns) == [
        "dataset_id",
        "source_file",
        "tune_position",
        "marker_name",
        "freq_target_ghz",
        "freq_ghz",
        "s_phase_deg",
    ]
    assert list(zip(result["marker_name"], result["tune_position"], strict=True)) == [
        ("f_2pi3", 1.0),
        ("f_2pi3", 2.0),
        ("f_mean", 1.0),
        ("f_mean", 2.0),
        ("f_pi2", 1.0),
        ("f_pi2", 2.0),
    ]
    assert result["s_phase_deg"].tolist() == pytest.approx(
        [
            -165.2797634916735,
            74.40030955943195,
            -178.74461937073156,
            26.81148233359222,
            169.48012254940267,
            -35.29997402642941,
        ]
    )


def test_build_position_phase_target_table_uses_polar_ideal_points() -> None:
    result = build_position_phase_target_table(
        _marker_points(),
        positions=(1.0, 2.0),
    )

    assert result["phase_0to360_deg"].tolist() == pytest.approx(
        [
            194.7202365083265,
            74.40030955943195,
            181.25538062926844,
            26.81148233359222,
            169.48012254940267,
            324.7000259735706,
        ]
    )
    assert result["ideal_phase_deg"].tolist() == pytest.approx(
        [180.0, 60.0, 180.0, 0.0, 180.0, 300.0]
    )
    assert result["phase_error_deg"].tolist() == pytest.approx(
        [
            -14.720236508326504,
            -14.400309559431954,
            -1.2553806292684435,
            -26.81148233359222,
            10.51987745059733,
            -24.700025973570582,
        ]
    )

    position_2 = result[result["tune_position"] == 2.0]
    assert position_2["phase_advance_0to360_deg"].tolist() == pytest.approx(
        [239.68007305110545, 205.55610170432377, 155.21990342416793]
    )
    assert position_2["ideal_phase_advance_deg"].tolist() == pytest.approx(
        [240.0, 180.0, 120.0]
    )
    assert position_2["phase_advance_error_deg"].tolist() == pytest.approx(
        [0.31992694889455, -25.55610170432377, -35.21990342416793]
    )


def test_phase_advance_difference_classifies_excess_and_deficit() -> None:
    assert POSITION_IDEAL_COLOR == ADVANCE_IDEAL_COLOR == "#d9dde3"
    assert ADVANCE_IDEAL_BAR_WIDTH == ADVANCE_MEASURED_BAR_WIDTH
    kind, magnitude = classify_phase_advance_difference(239.680073, 240.0)
    assert kind == "deficit"
    assert magnitude == pytest.approx(0.319927)

    kind, magnitude = classify_phase_advance_difference(205.556102, 180.0)
    assert kind == "excess"
    assert magnitude == pytest.approx(25.556102)

    kind, magnitude = classify_phase_advance_difference(120.0, 120.0)
    assert kind == "matched"
    assert magnitude == pytest.approx(0.0)


def test_build_position_phase_bar_table_rejects_missing_pair() -> None:
    marker_points = _marker_points().iloc[:-1].copy()

    with pytest.raises(
        ValueError,
        match=r"expected exactly one row for marker 'f_pi2' at position 2.0; found 0",
    ):
        build_position_phase_bar_table(marker_points, positions=(1.0, 2.0))


def test_build_position_phase_bar_table_rejects_duplicate_pair() -> None:
    marker_points = _marker_points()
    duplicate = marker_points[
        (marker_points["marker_name"] == "f_mean")
        & (marker_points["tune_position"] == 1.0)
    ]
    marker_points = pd.concat([marker_points, duplicate], ignore_index=True)

    with pytest.raises(
        ValueError,
        match=r"expected exactly one row for marker 'f_mean' at position 1.0; found 2",
    ):
        build_position_phase_bar_table(marker_points, positions=(1.0, 2.0))


def test_plot_position_phase_bars_writes_png_and_closes_figure(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "position_phase_bars.png"
    figures_before = set(plt.get_fignums())

    result = plot_position_phase_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert set(plt.get_fignums()) == figures_before


def test_plot_position_phase_advance_bars_writes_separate_png(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "position_phase_advance_bars.png"
    figures_before = set(plt.get_fignums())

    result = plot_position_phase_advance_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert set(plt.get_fignums()) == figures_before


def test_position_phase_figures_omit_legends(tmp_path: Path) -> None:
    phase_path = tmp_path / "position_phase.svg"
    advance_path = tmp_path / "phase_advance.svg"

    plot_position_phase_bars(
        _marker_points(),
        phase_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )
    plot_position_phase_advance_bars(
        _marker_points(),
        advance_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    assert 'id="legend_' not in phase_path.read_text(encoding="utf-8")
    assert 'id="legend_' not in advance_path.read_text(encoding="utf-8")


def test_position_phase_bars_annotate_ideal_minus_measured_error(tmp_path: Path) -> None:
    output_path = tmp_path / "position_phase.svg"

    plot_position_phase_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    svg = output_path.read_text(encoding="utf-8")
    assert "<!-- -14.7° -->" in svg
    assert "<!-- +10.5° -->" in svg
    assert "<!-- -26.8° -->" in svg


def test_position_phase_bars_place_small_error_labels_to_the_right(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "position_phase.svg"

    plot_position_phase_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    svg = output_path.read_text(encoding="utf-8")
    error_x = _svg_text_translate_x(svg, "-14.7°")
    measured_x = _svg_text_translate_x(svg, "194.7°")

    assert error_x > measured_x
    assert error_x - measured_x < 90.0


def test_position_phase_bars_place_pi2_position_1_error_to_the_left(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "position_phase.svg"

    plot_position_phase_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    svg = output_path.read_text(encoding="utf-8")
    pi2_error_x = _svg_text_translate_x(svg, "+10.5°")
    pi2_measured_x = _svg_text_translate_x(svg, "169.5°")

    assert pi2_error_x < pi2_measured_x
    assert pi2_measured_x - pi2_error_x < 90.0


def test_position_phase_bars_enlarge_frequency_mode_labels(tmp_path: Path) -> None:
    output_path = tmp_path / "position_phase.svg"

    plot_position_phase_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    svg = output_path.read_text(encoding="utf-8")

    assert re.search(
        r"<!-- \$f_\{2\\pi/3\}\$ -->\s*<g transform=.*scale\(0\.17 -0\.17\)",
        svg,
    )


def test_position_phase_advance_bars_annotate_ideal_minus_measured_error(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "phase_advance.svg"

    plot_position_phase_advance_bars(
        _marker_points(),
        output_path,
        positions=(1.0, 2.0),
        config=PlotConfig(dpi=90),
    )

    svg = output_path.read_text(encoding="utf-8")
    assert "<!-- +0.3° -->" in svg
    assert "<!-- -25.6° -->" in svg


def _svg_text_translate_x(svg: str, label: str) -> float:
    match = re.search(
        rf"<!-- {re.escape(label)} -->\s*<g[^>]*transform=\"translate\(([-0-9.]+)",
        svg,
    )
    assert match is not None
    return float(match.group(1))
