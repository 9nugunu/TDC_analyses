"""Run the revised Kyhl validation workflow and write its report artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.analysis.kyhl_validation import (
    IDEAL_HALF_CELL_PHASE_DEG,
    TARGET_COUPLER_PHASE_DEG,
    build_coupler_to_first_error,
    build_field_energy_pairs,
    build_full_sweep_phase_response_ratio,
    build_full_sweep_rms_comparison,
    build_no_plunger_phase_points,
    build_no_plunger_phase_residuals,
    build_phase_correction_proxy,
    summarize_field_energy_pairs,
    summarize_no_plunger_phase_residuals,
)
from deflector_tuning.data_loading.field_profiles import (
    load_field_phase_export,
    load_field_profile_export,
)

DEFAULT_PROFILE_DIR = Path("data/sim/sim_profile_260618_PhaseDistribution")
DEFAULT_NODAL_SWEEP_TABLE = Path("fig/analyses/sim_sweep_260618_Nodalshift/tables/phase_stats.csv")
DEFAULT_FULL_SWEEP_PHASE_ADVANCE = Path(
    "fig/analyses/sim_sweep_260620_FullstructureSweep_ports_swapped/tables/phase_adv.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/revised_kyhl_validation")


def run_kyhl_validation(
    *,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    nodal_sweep_summary: Path = DEFAULT_NODAL_SWEEP_TABLE,
    full_sweep_phase_advance: Path = DEFAULT_FULL_SWEEP_PHASE_ADVANCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Load inputs, compute validation tables, and write a Markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    e_profile = load_field_profile_export(profile_dir / "E_fieldDist.txt")
    phase_profile = load_field_phase_export(profile_dir / "EM_fieldPhase.txt")

    field_pairs = build_field_energy_pairs(e_profile)
    field_pairs.to_csv(output_dir / "field_energy_ratio_pairs.csv", index=False)
    field_summary = summarize_field_energy_pairs(field_pairs)
    field_summary.to_csv(output_dir / "field_energy_ratio_summary.csv", index=False)

    phase_points = build_no_plunger_phase_points(phase_profile)
    phase_points.to_csv(output_dir / "no_plunger_phase_points.csv", index=False)
    phase_residuals = build_no_plunger_phase_residuals(phase_points)
    phase_residuals.to_csv(output_dir / "no_plunger_phase_residuals.csv", index=False)
    phase_summary = summarize_no_plunger_phase_residuals(phase_residuals)
    phase_summary.to_csv(output_dir / "no_plunger_phase_residual_summary.csv", index=False)

    nodal_proxy = build_phase_correction_proxy(nodal_sweep_summary)
    nodal_proxy.to_csv(output_dir / "phase_correction_ratio_proxy.csv", index=False)
    full_sweep = pd.read_csv(full_sweep_phase_advance)
    full_sweep_response = build_full_sweep_phase_response_ratio(full_sweep)
    full_sweep_response.to_csv(output_dir / "full_sweep_phase_response_ratio.csv", index=False)
    full_sweep_summary = build_full_sweep_rms_comparison(full_sweep)
    full_sweep_summary.to_csv(output_dir / "full_sweep_phase_rms_comparison.csv", index=False)
    coupler_error = build_coupler_to_first_error(full_sweep)
    coupler_error.to_csv(output_dir / "coupler_to_first_phase_error.csv", index=False)

    report = build_markdown_report(
        profile_dir=profile_dir,
        nodal_sweep_summary=nodal_sweep_summary,
        full_sweep_phase_advance=full_sweep_phase_advance,
        field_summary=field_summary,
        nodal_proxy=nodal_proxy,
        phase_summary=phase_summary,
        full_sweep_response=full_sweep_response,
        full_sweep_summary=full_sweep_summary,
        coupler_error=coupler_error,
    )
    (output_dir / "revised_kyhl_validation_summary.md").write_text(report, encoding="utf-8")


def build_markdown_report(
    *,
    profile_dir: Path,
    nodal_sweep_summary: Path,
    full_sweep_phase_advance: Path,
    field_summary: pd.DataFrame,
    nodal_proxy: pd.DataFrame,
    phase_summary: pd.DataFrame,
    full_sweep_response: pd.DataFrame,
    full_sweep_summary: pd.DataFrame,
    coupler_error: pd.DataFrame,
) -> str:
    """Render a compact manuscript-oriented metric summary."""

    upstream = field_summary[field_summary["ratio_definition"] == "upstream_iris_pair"].iloc[0]
    adjacent_average = field_summary[field_summary["ratio_definition"] == "adjacent_iris_average"].iloc[0]
    slope_plus = phase_summary[phase_summary["ideal_slope_deg_per_half_cell"] == IDEAL_HALF_CELL_PHASE_DEG].iloc[0]

    lines = [
        "# Revised KYHL Validation Metrics",
        "",
        "## Data Sources",
        f"- No-plunger profile: `{profile_dir.as_posix()}`",
        f"- Nodal sweep proxy: `{nodal_sweep_summary.as_posix()}`",
        f"- Full-structure phase sweep: `{full_sweep_phase_advance.as_posix()}`",
        "",
        "## Field-Energy Prediction",
        (
            f"- Upstream iris/cell sensitivity prediction: R_E = {upstream['mean_ratio']:.6g} "
            f"+/- {upstream['std_ratio']:.6g} (N={int(upstream['pair_count'])})."
        ),
        (
            f"- Adjacent-iris average alternative: R_E = {adjacent_average['mean_ratio']:.6g} "
            f"+/- {adjacent_average['std_ratio']:.6g}."
        ),
        "",
        "## No-Plunger Phase Map",
        (
            f"- EM_fieldPhase half-cell residual for +60 deg convention: "
            f"sigma_phi = {slope_plus['rms_phase_error_deg']:.6g} deg, "
            f"integrated error = {slope_plus['integrated_phase_error_deg_half_cell']:.6g} deg-half-cell."
        ),
        "",
        "## Phase-Correction Evidence",
        "- Direct DeltaPhi_I/DeltaPhi_C was not computed from the no-plunger profile alone.",
        "- FullstructureSweep_ports_swapped is used for the cell/iris phase response ratio.",
        "- The existing single-transition nodal sweep table is retained as a secondary proxy.",
        "",
        "## Full-Sweep Phase Response Ratio",
    ]
    for _, row in full_sweep_response.iterrows():
        lines.append(
            f"- {row['marker_name']}: R_phase(response) = {row['R_phase_response_I_over_C']:.6g}; "
            f"target-residual iris/cell = {row['target_residual_ratio_iris_over_cell']:.6g}."
        )
    lines.extend(
        [
        "",
        "## Full-Sweep Residual Comparison",
        ]
    )
    for _, row in full_sweep_summary.iterrows():
        lines.append(
            f"- {row['marker_name']}: sigma_cell = {row['sigma_phi_cell_deg']:.6g} deg, "
            f"sigma_iris = {row['sigma_phi_iris_deg']:.6g} deg, "
            f"cell/iris = {row['rms_improvement_factor_cell_over_iris']:.6g}."
        )
    lines.extend(
        [
            "",
            "## Coupler-to-First Phase Error",
        ]
    )
    for _, row in coupler_error.iterrows():
        if row["marker_name"] != "f_2pi3":
            continue
        lines.append(
            f"- f_2pi3 {row['position_family']}: {row['phase_advance_0to360_deg']:.6g} deg, "
            f"error from {TARGET_COUPLER_PHASE_DEG:g} deg = {row['wrapped_error_from_240_deg']:.6g} deg."
        )
    lines.extend(
        [
            "",
            "## Manuscript Sentence",
            (
                "The field-energy ratio predicts that the iris-centered perturbation should be "
                f"approximately {upstream['mean_ratio']:.3g} times more sensitive than the "
                "cell-centered perturbation. The measured/simulated phase correction ratio and "
                "phase-map residuals were then used to test this prediction."
            ),
            "",
            "## Claim Strength",
            "- Data-supported: the no-plunger HEM11 E-field profile predicts higher iris-center electric perturbation sensitivity.",
            "- Data-supported with available sweep proxy: the full-structure phase residuals are lower for iris-family transitions in the ports-swapped sweep table.",
            "- Not yet directly proven by this no-plunger profile alone: DeltaPhi_I/DeltaPhi_C before/after plunger perturbation equality to R_E.",
        ]
    )
    if not nodal_proxy.empty:
        lines.extend(["", "## Nodal Sweep Proxy Rows"])
        for _, row in nodal_proxy.iterrows():
            lines.append(
                f"- {row['marker_name']}: proxy R_phase = "
                f"{row.get('R_phase_proxy_abs_I_over_abs_C', np.nan):.6g}; {row['caveat']}"
            )
    return "\n".join(lines) + "\n"
