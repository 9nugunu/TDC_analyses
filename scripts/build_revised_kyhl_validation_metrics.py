"""Build revised KYHL validation metrics from profile and phase tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-export analysis functions for existing imports; the CLI only handles arguments.
from deflector_tuning.analysis.kyhl_validation import (  # noqa: E402
    IDEAL_HALF_CELL_PHASE_DEG,
    REGULAR_CELL_COUNT,
    TARGET_COUPLER_PHASE_DEG,
    HalfCellStation,
    _normalized_field_kind,
    _single_phase_trace,
    _single_profile_trace,
    _sorted_xy,
    build_coupler_to_first_error,
    build_field_energy_pairs,
    build_full_sweep_phase_response_ratio,
    build_full_sweep_rms_comparison,
    build_half_cell_stations,
    build_no_plunger_phase_points,
    build_no_plunger_phase_residuals,
    build_phase_correction_proxy,
    interpolate_profile,
    rms,
    safe_ratio,
    summarize_field_energy_pairs,
    summarize_no_plunger_phase_residuals,
    wrap_deg,
)
from deflector_tuning.data_loading.field_profiles import (  # noqa: E402
    FieldPhaseExport,
    FieldProfileExport,
    load_field_phase_export,
    load_field_profile_export,
)
from deflector_tuning.workflows.kyhl_validation import (  # noqa: E402
    DEFAULT_FULL_SWEEP_PHASE_ADVANCE,
    DEFAULT_NODAL_SWEEP_TABLE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROFILE_DIR,
    build_markdown_report,
    run_kyhl_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--nodal-sweep-summary", type=Path, default=DEFAULT_NODAL_SWEEP_TABLE)
    parser.add_argument("--full-sweep-phase-advance", type=Path, default=DEFAULT_FULL_SWEEP_PHASE_ADVANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_kyhl_validation(
        profile_dir=args.profile_dir,
        nodal_sweep_summary=args.nodal_sweep_summary,
        full_sweep_phase_advance=args.full_sweep_phase_advance,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
