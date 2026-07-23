"""Project first/second iris raw phase changes onto a simulated ``r_c`` scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.analysis.phase_radius_equivalence import estimate_phase_radius_equivalence
from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.visualization.phase_radius_equivalence_plots import plot_phase_radius_equivalence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_phase_line_csv", type=Path, help="Simulation r_c line-scan CSV")
    parser.add_argument("sim_markers_csv", type=Path, help="Simulation marker-frequency CSV")
    parser.add_argument("before_folder", type=Path, help="Raw folder before tuning")
    parser.add_argument("after_folder", type=Path, help="Raw folder after tuning")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-r-c-mm", type=float, default=56.59)
    parser.add_argument("--fixed-w-c-mm", type=float, default=19.3224)
    parser.add_argument("--min-delta-r-c-um", type=float, default=-20.0)
    parser.add_argument("--max-delta-r-c-um", type=float, default=120.0)
    args = parser.parse_args()

    line = pd.read_csv(args.sim_phase_line_csv)
    markers = pd.read_csv(args.sim_markers_csv).loc[:, ["marker_name", "freq_ghz"]]
    projections: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    raw_rows: list[pd.DataFrame] = []
    estimate_rows: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []
    for position, iris_label in ((1, "First iris (port 1)"), (2, "Second iris (port 2)")):
        raw = _raw_phase_change(args.before_folder / f"{position}_portE.S2P", args.after_folder / f"{position}_portE.S2P", markers)
        estimate, curves = estimate_phase_radius_equivalence(
            line,
            raw.loc[:, ["marker_name", "phase_delta_deg"]],
            baseline_r_c_mm=args.baseline_r_c_mm,
            min_delta_r_c_um=args.min_delta_r_c_um,
            max_delta_r_c_um=args.max_delta_r_c_um,
        )
        projections[iris_label] = (estimate, curves)
        raw_rows.append(raw.assign(iris=iris_label))
        estimate_rows.append(estimate.assign(iris=iris_label))
        curve_rows.append(curves.assign(iris=iris_label))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_table = pd.concat(raw_rows, ignore_index=True)
    estimate_table = pd.concat(estimate_rows, ignore_index=True)
    curve_table = pd.concat(curve_rows, ignore_index=True)
    raw_table.to_csv(args.output_dir / "raw_phase_change_at_simulation_markers.csv", index=False)
    estimate_table.to_csv(args.output_dir / "equivalent_delta_r_c_estimates.csv", index=False)
    curve_table.to_csv(args.output_dir / "simulation_phase_change_curves.csv", index=False)
    path = plot_phase_radius_equivalence(
        projections,
        args.output_dir / "raw_tuning_projected_to_simulation_r_c.png",
        baseline_r_c_mm=args.baseline_r_c_mm,
        fixed_w_c_mm=args.fixed_w_c_mm,
    )
    print(path)


def _raw_phase_change(before_path: Path, after_path: Path, markers: pd.DataFrame) -> pd.DataFrame:
    before = _sample_s11(before_path, markers)
    after = _sample_s11(after_path, markers)
    output = before.merge(after, on=["marker_name", "freq_ghz"], suffixes=("_before", "_after"))
    output["phase_delta_deg"] = np.angle(
        output["s11_after"].to_numpy() * np.conj(output["s11_before"].to_numpy()),
        deg=True,
    )
    return output.drop(columns=["s11_before", "s11_after"])


def _sample_s11(path: Path, markers: pd.DataFrame) -> pd.DataFrame:
    data = read_touchstone(path)
    frequency_hz = np.asarray(data.frequency, dtype=float)
    s11 = np.asarray([row[0] for row in data.s_values])
    targets_hz = markers["freq_ghz"].to_numpy(dtype=float) * 1e9
    sampled = np.interp(targets_hz, frequency_hz, s11.real) + 1j * np.interp(targets_hz, frequency_hz, s11.imag)
    return markers.assign(s11_phase_deg=np.angle(sampled, deg=True), s11=sampled).drop(columns="s11_phase_deg")


if __name__ == "__main__":
    main()
