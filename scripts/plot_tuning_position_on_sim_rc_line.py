"""Locate a measured raw state on the simulated port-2 ``r_c`` phase line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS
from deflector_tuning.analysis.phase_radius_equivalence import build_raw_anchored_phase_position
from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.analysis.complex_sampling import sample_complex_frequency
from deflector_tuning.visualization.phase_radius_equivalence_plots import plot_raw_phase_position_on_simulation_line


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options using the current project scientific defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_phase_line_csv", type=Path, help="Simulation r_c line-scan CSV for port 2")
    parser.add_argument("sim_markers_csv", type=Path, help="Simulation marker-frequency CSV")
    parser.add_argument("before_folder", type=Path, help="Raw folder before tuning")
    parser.add_argument("after_folder", type=Path, help="Raw folder after tuning")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-r-c-mm", type=float, default=float(DEFAULT_PROJECT_DEFAULTS.design_point_by_axis["sim_r_c"]))
    parser.add_argument("--target-r-c-mm", type=float, default=float(DEFAULT_PROJECT_DEFAULTS.design_point_by_axis["sim_r_c"]), help="Simulation design/reference value")
    parser.add_argument("--fixed-w-c-mm", type=float, default=float(DEFAULT_PROJECT_DEFAULTS.design_point_by_axis["sim_w_c"]))
    parser.add_argument("--state-label", default="Measured state", help="Measured-state display label")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    markers = pd.read_csv(args.sim_markers_csv).loc[:, ["marker_name", "freq_ghz"]]
    first_iris = _raw_phase_observation(
        args.before_folder / "1_portE.S2P",
        args.after_folder / "1_portE.S2P",
        markers,
    )
    second_iris = _raw_phase_observation(
        args.before_folder / "2_portE.S2P",
        args.after_folder / "2_portE.S2P",
        markers,
    )
    position, curves = build_raw_anchored_phase_position(
        pd.read_csv(args.sim_phase_line_csv),
        second_iris,
        baseline_r_c_mm=args.baseline_r_c_mm,
        target_r_c_mm=args.target_r_c_mm,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_iris.to_csv(args.output_dir / "first_iris_raw_phase_observation.csv", index=False)
    second_iris.to_csv(args.output_dir / "second_iris_raw_phase_observation.csv", index=False)
    position.to_csv(args.output_dir / "second_iris_current_position.csv", index=False)
    curves.to_csv(args.output_dir / "second_iris_raw_anchored_simulation_curves.csv", index=False)
    path = plot_raw_phase_position_on_simulation_line(
        position,
        curves,
        first_iris,
        args.output_dir / "torque13p5_position_on_sparameter_phase_vs_r_c.png",
        baseline_r_c_mm=args.baseline_r_c_mm,
        target_r_c_mm=args.target_r_c_mm,
        fixed_w_c_mm=args.fixed_w_c_mm,
        measured_state_label=args.state_label,
    )
    print(path)


def _raw_phase_observation(before_path: Path, after_path: Path, markers: pd.DataFrame) -> pd.DataFrame:
    before = _sample_s11_phase(before_path, markers).rename(columns={"s11_phase_deg": "before_phase_deg"})
    after = _sample_s11_phase(after_path, markers).rename(columns={"s11_phase_deg": "after_phase_deg"})
    observation = before.merge(after, on=["marker_name", "freq_ghz"], validate="one_to_one")
    observation["phase_delta_deg"] = (observation["after_phase_deg"] - observation["before_phase_deg"] + 180.0) % 360.0 - 180.0
    return observation


def _sample_s11_phase(path: Path, markers: pd.DataFrame) -> pd.DataFrame:
    data = read_touchstone(path)
    if data.header.parameter != "S":
        raise ValueError("Phase sampling requires S-parameter data")
    sampled = sample_complex_frequency(
        data.frequency, np.asarray(data.s_values)[:, 0], markers["freq_ghz"].to_numpy(dtype=float),
        frequency_unit=data.header.frequency_unit,
    )
    return markers.assign(s11_phase_deg=np.angle(sampled, deg=True))


if __name__ == "__main__":
    main()
