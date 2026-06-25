import numpy as np
import pytest

from deflector_tuning.analysis.field_energy_ratio import (
    compute_cell_iris_field_energy_ratios,
    summarize_cell_iris_field_energy_ratios,
)
from deflector_tuning.visualization.em_field_structure_plots import FieldProfileExport, FieldProfileTrace


def test_compute_cell_iris_field_energy_ratios_samples_cell_and_adjacent_irises() -> None:
    export = FieldProfileExport(
        parameters={"d": 4.0, "t": 2.0},
        traces=(
            FieldProfileTrace(
                label="E field",
                field_kind="e",
                component="y",
                value_kind="real",
                z_mm=np.array([1.0, 4.0, 7.0]),
                values=np.array([2.0, 1.0, 4.0]),
            ),
        ),
    )

    ratios = compute_cell_iris_field_energy_ratios(export, regular_cell_count=1)
    summary = summarize_cell_iris_field_energy_ratios(ratios)

    row = ratios.iloc[0]
    assert row["cell_label"] == "R1"
    assert row["upstream_iris_label"] == "IN"
    assert row["downstream_iris_label"] == "OUT"
    assert row["cell_center_z_mm"] == pytest.approx(4.0)
    assert row["upstream_iris_to_cell_energy_ratio"] == pytest.approx(4.0)
    assert row["downstream_iris_to_cell_energy_ratio"] == pytest.approx(16.0)
    assert row["adjacent_average_iris_to_cell_energy_ratio"] == pytest.approx(10.0)
    assert summary.loc[0, "ratio_definition"] == "upstream_iris_to_cell"
    assert summary.loc[0, "mean_energy_ratio"] == pytest.approx(4.0)
