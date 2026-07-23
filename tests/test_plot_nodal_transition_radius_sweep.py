from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.plot_nodal_transition_radius_sweep import (
    _draw_transition_sweep,
    build_transition_sweep,
    plot_transition_sweep,
)


def _reference_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "reference",
                "marker_name": "f_2pi3",
                "tune_position": 3.5,
                "s_phase_deg": -170.0,
                "source_file": "cell_reference.s2p",
            },
            {
                "dataset_id": "reference",
                "marker_name": "f_2pi3",
                "tune_position": 4.0,
                "s_phase_deg": -110.0,
                "source_file": "iris_reference.s2p",
            },
            {
                "dataset_id": "reference",
                "marker_name": "f_pi2",
                "tune_position": 3.5,
                "s_phase_deg": 20.0,
                "source_file": "ignored.s2p",
            },
        ]
    )


def _sweep_points() -> pd.DataFrame:
    rows = []
    for offset, cell_phase, iris_phase in [
        (-0.1, 60.0, 120.0),
        (0.0, 70.0, 130.0),
        (0.1, 80.0, 140.0),
    ]:
        rows.extend(
            [
                {
                    "dataset_id": "fine",
                    "marker_name": "f_2pi3",
                    "tune_position": 4.5,
                    "sim_offset_cell_03": offset,
                    "s_phase_deg": cell_phase,
                    "source_file": f"cell_{offset}.s2p",
                },
                {
                    "dataset_id": "fine",
                    "marker_name": "f_2pi3",
                    "tune_position": 5.0,
                    "sim_offset_cell_03": offset,
                    "s_phase_deg": iris_phase,
                    "source_file": f"iris_{offset}.s2p",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_build_transition_sweep_computes_same_family_forward_wrapped_advances() -> None:
    result = build_transition_sweep(
        _reference_points(),
        _sweep_points(),
        base_radius_mm=57.09,
    )

    assert len(result) == 6
    assert set(result["marker_name"]) == {"f_2pi3"}
    assert set(result["family"]) == {"cell", "iris"}
    assert result["offset_cell_03_mm"].tolist() == [-0.1, -0.1, 0.0, 0.0, 0.1, 0.1]
    assert result["r_c_mm"].tolist() == pytest.approx([56.99, 56.99, 57.09, 57.09, 57.19, 57.19])

    zero = result[result["offset_cell_03_mm"] == 0.0].set_index("family")
    assert zero.loc["cell", "pos_from"] == 3.5
    assert zero.loc["cell", "pos_to"] == 4.5
    assert zero.loc["cell", "phase_adv_deg"] == pytest.approx(240.0)
    assert zero.loc["iris", "pos_from"] == 4.0
    assert zero.loc["iris", "pos_to"] == 5.0
    assert zero.loc["iris", "phase_adv_deg"] == pytest.approx(240.0)


def test_plot_transition_sweep_uses_family_colors_target_and_zero_annotations(tmp_path: Path) -> None:
    table = build_transition_sweep(
        _reference_points(),
        _sweep_points(),
        base_radius_mm=57.09,
    )

    path = plot_transition_sweep(table, tmp_path / "transition.png", dpi=120)

    assert path.exists()
    fig, ax = _draw_transition_sweep(table)
    family_lines = [line for line in ax.lines if line.get_label() in {"Cell 3.5→4.5", "Iris 4.0→5.0"}]
    assert len(family_lines) == 2
    assert family_lines[0].get_color() != family_lines[1].get_color()
    assert any(line.get_label() == "Ideal 240°" for line in ax.lines)
    assert {text.get_text() for text in ax.texts} == {
        "Cell: 240.0°",
        "Iris: 240.0°",
    }
    plt.close(fig)
