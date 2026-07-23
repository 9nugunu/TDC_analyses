from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.plot_position_phase_bars import main


def test_main_writes_position_phase_bar_png_and_csv(tmp_path: Path) -> None:
    marker_path = tmp_path / "marker_pts.csv"
    output_path = tmp_path / "figures" / "position_phase_bars.png"
    rows = []
    for marker_name, frequency, phase_1, phase_2 in [
        ("f_2pi3", 2.856, -165.2, 74.4),
        ("f_mean", 2.866, -178.7, 26.8),
        ("f_pi2", 2.876, 169.5, -35.3),
    ]:
        for position, phase in [(1.0, phase_1), (2.0, phase_2)]:
            rows.append(
                {
                    "dataset_id": "raw_sweep_sample",
                    "source_file": f"{position:g}_portE.S2P",
                    "tune_position": position,
                    "marker_name": marker_name,
                    "freq_target_ghz": frequency,
                    "freq_ghz": frequency - 0.000007,
                    "s_phase_deg": phase,
                }
            )
    pd.DataFrame(rows).to_csv(marker_path, index=False)

    result = main(
        [
            "--marker-points",
            str(marker_path),
            "--positions",
            "1.0",
            "2.0",
            "--output",
            str(output_path),
        ]
    )

    assert result == output_path
    assert output_path.exists()
    advance_output = (
        output_path.parent / "position_phase_bars_phase_advance.png"
    )
    assert advance_output.exists()
    sidecar = output_path.with_suffix(".csv")
    assert sidecar.exists()
    comparison = pd.read_csv(sidecar)
    assert list(zip(comparison["marker_name"], comparison["tune_position"], strict=True)) == [
        ("f_2pi3", 1.0),
        ("f_2pi3", 2.0),
        ("f_mean", 1.0),
        ("f_mean", 2.0),
        ("f_pi2", 1.0),
        ("f_pi2", 2.0),
    ]
    assert comparison["ideal_phase_deg"].tolist() == [
        180.0,
        60.0,
        180.0,
        0.0,
        180.0,
        300.0,
    ]
    assert comparison.loc[
        comparison["tune_position"] == 2.0,
        "ideal_phase_advance_deg",
    ].tolist() == [240.0, 180.0, 120.0]
