from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.plot_wc_absolute_s11_phase import (
    build_wc_absolute_phase_table,
    draw_wc_absolute_phase,
    plot_wc_absolute_phase,
)


def _phase_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker_name": ["f_2pi3"] * 4 + ["f_mean"],
            "sim_NumDepth": [2.0] * 5,
            "sim_r_c": [56.59, 56.59, 56.59, 56.60, 56.59],
            "sim_w_c": [19.5724, 19.3224, 19.0724, 19.3224, 19.3224],
            "s_phase_deg": [52.18, 61.98, 72.04, 10.0, 20.0],
            "delta_phase_reference_w_c_mm": [19.3224] * 5,
        }
    )


def test_build_wc_absolute_phase_table_keeps_only_requested_iris_scan() -> None:
    table = build_wc_absolute_phase_table(
        _phase_points(),
        fixed_r_c_mm=56.59,
        num_depth=2.0,
    )

    assert table["sim_w_c"].tolist() == pytest.approx([19.0724, 19.3224, 19.5724])
    assert table["s_phase_deg"].tolist() == pytest.approx([72.04, 61.98, 52.18])
    assert set(table["marker_name"]) == {"f_2pi3"}


def test_draw_wc_absolute_phase_uses_absolute_s11_values() -> None:
    table = build_wc_absolute_phase_table(
        _phase_points(),
        fixed_r_c_mm=56.59,
        num_depth=2.0,
    )

    fig, ax = draw_wc_absolute_phase(table, fixed_r_c_mm=56.59)
    try:
        iris_line = next(line for line in ax.lines if line.get_label() == r"$f_{2\pi/3}$ Iris")
        assert list(iris_line.get_ydata()) == pytest.approx([72.04, 61.98, 52.18])
        assert r"\phi" in ax.get_ylabel()
        assert r"S" in ax.get_ylabel()
        assert r"\Delta" not in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_plot_wc_absolute_phase_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "absolute.png"
    table = build_wc_absolute_phase_table(
        _phase_points(),
        fixed_r_c_mm=56.59,
        num_depth=2.0,
    )

    result = plot_wc_absolute_phase(
        table,
        output_path,
        fixed_r_c_mm=56.59,
    )

    assert result == output_path
    assert output_path.is_file()
