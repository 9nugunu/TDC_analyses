"""Build KYHL operation-mode admittance transition metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deflector_tuning.analysis.kyhl_admittance import (  # noqa: E402
    compute_kyhl_admittance_points,
    compute_kyhl_admittance_transitions,
)
from deflector_tuning.analysis.sparameter_selection import select_s11_rows  # noqa: E402
from deflector_tuning.data_loading.central_loader import DataLoader  # noqa: E402
from deflector_tuning.markers.frequency_markers import extract_marker_frequencies  # noqa: E402
from deflector_tuning.visualization.kyhl_admittance_plots import plot_kyhl_operation_polar  # noqa: E402


DEFAULT_SPARAMETER_PATH = Path("data/sim/sim_sweep_260620_Halfstructure")
DEFAULT_DISPERSION_PATH = Path("data/sim/sim_dispersion_260505_single_cell_step1")
DEFAULT_OUTPUT_DIR = Path("outputs/kyhl_admittance_operation_metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparameter-path", type=Path, default=DEFAULT_SPARAMETER_PATH)
    parser.add_argument("--dispersion-path", type=Path, default=DEFAULT_DISPERSION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--marker-role", default="sim")
    parser.add_argument("--operation-mode-deg", type=float, default=120.0)
    parser.add_argument("--axis-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--file-workers", type=int, default=1)
    parser.add_argument("--plot-marker", default="f_2pi3")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_points = build_marker_points_with_complex_samples(
        sparameter_path=args.sparameter_path,
        dispersion_path=args.dispersion_path,
        marker_role=args.marker_role,
        file_workers=args.file_workers,
    )
    marker_points.to_csv(output_dir / "kyhl_marker_points_with_complex.csv", index=False)

    operation_points = compute_kyhl_admittance_points(
        marker_points,
        operation_mode_deg=args.operation_mode_deg,
        axis_sign=args.axis_sign,
    )
    operation_points.to_csv(output_dir / "kyhl_admittance_points.csv", index=False)

    transitions = compute_kyhl_admittance_transitions(
        marker_points,
        operation_mode_deg=args.operation_mode_deg,
        axis_sign=args.axis_sign,
    )
    transitions.to_csv(output_dir / "kyhl_admittance_transitions.csv", index=False)

    coupler_check = build_coupler_to_first_check(transitions)
    coupler_check.to_csv(output_dir / "kyhl_coupler_to_first_check.csv", index=False)

    figure_dir = output_dir / "figures"
    plot_kyhl_operation_polar(
        operation_points,
        transitions,
        figure_dir / f"kyhl_operation_{args.plot_marker}_polar.png",
        marker_name=args.plot_marker,
    )


def build_marker_points_with_complex_samples(
    *,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
    file_workers: int = 1,
) -> pd.DataFrame:
    """Sample marker frequencies while preserving raw complex S11 columns."""

    loader = DataLoader(file_workers=file_workers)
    sparameter_table = select_s11_rows(loader.load(sparameter_path))
    markers = extract_marker_frequencies(dispersion_path, marker_role=marker_role)
    frames = [_sample_one_marker_with_complex(sparameter_table, marker) for _, marker in markers.iterrows()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_coupler_to_first_check(transitions: pd.DataFrame) -> pd.DataFrame:
    """Return the first transition per marker and position family."""

    if transitions.empty:
        return transitions.copy()
    rows = []
    for (_marker_name, _family), group in transitions.groupby(["marker_name", "position_family"], sort=False):
        rows.append(group.sort_values(["pos_from", "pos_to"], kind="mergesort").iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _sample_one_marker_with_complex(sparameter_table: pd.DataFrame, marker: pd.Series) -> pd.DataFrame:
    target_freq_ghz = float(marker["freq_ghz"])
    table = sparameter_table.copy()
    table["freq_target_ghz"] = target_freq_ghz
    table["freq_error_ghz"] = table["freq_ghz"] - target_freq_ghz
    table["_abs_freq_error_ghz"] = table["freq_error_ghz"].abs()
    for column in ["source_file", "s_name", "tune_position", "port_side"]:
        if column not in table:
            table[column] = pd.NA

    nearest_index = table.groupby(["source_file", "s_name", "tune_position", "port_side"], dropna=False)[
        "_abs_freq_error_ghz"
    ].idxmin()
    nearest = table.loc[nearest_index].copy()
    nearest["marker_name"] = marker["marker_name"]
    nearest["marker_role"] = marker["marker_role"]
    nearest["marker_source"] = marker["marker_source"]
    for column in [
        "uncorrected_freq_ghz",
        "frequency_scale_factor",
        "temp_op_C",
        "temp_meas_C",
        "humidity_fraction",
    ]:
        nearest[column] = marker[column] if column in marker else pd.NA
    return nearest.drop(columns=["_abs_freq_error_ghz"])


if __name__ == "__main__":
    main()
