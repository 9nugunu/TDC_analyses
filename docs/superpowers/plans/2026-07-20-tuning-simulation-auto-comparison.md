# Tuning Experiment to Simulation Auto-Comparison Plan

> Status: implementation plan for the 2026-07-20 tuning campaign workflow.

## Goal

When `run_folder_analysis.py` analyzes an experimental dataset whose folder name
contains the exact `_tune_` token, automatically resolve its campaign metadata,
select the simulation reference for each measured geometry family, compare the
current state with the campaign baseline, and save the comparison tables and
figures under the tuned dataset's normal analysis output.

The result is a design-point-anchored *equivalent* `Delta r_c` response. It is
not a direct bolt-travel or absolute physical-radius measurement.

## Data contracts

- Dataset routing remains centralized by source root (`data/raw`, `data/sim`).
- No campaign-ID or dataset-name branches are allowed in Python code.
- Campaign YAML declares simulation references by measurement family. Automatic
  mapping primarily consumes a one-dimensional `r_c` sweep:

  ```yaml
  campaign:
    sims:
      iris:
        data: sim_grid_260701_1DRcFine
        exp_pos: [1, 2]
        axis: r_c
      cell:
        data: sim_grid_260720_1DRcFine_Cell
        exp_pos: [0.5, 1.5]
        axis: r_c
  ```

- Integer experimental positions are iris measurements; half-integer positions
  are cell measurements. The explicit `exp_pos` list remains authoritative for
  selecting the anchor and comparison locations.
- `axis: r_c` is required for automatic phase-to-radius projection. A 2-D grid
  is eligible only after the existing analysis has produced an unambiguous
  one-dimensional `r_c` line at a declared fixed value of the other axis; the
  comparison workflow never fits directly against a 2-D grid.
- The first configured position (iris `1`, cell `0.5`) is the reference-plane
  anchor. Its `f_mean` phase is checked against 180 degrees using wrapped phase.
- The second configured position (iris `2`, cell `1.5`) supplies the measured
  before/after phase change that is projected onto the matching simulation
  `r_c` response.
- Experimental marker frequencies already include the measurement temperature
  correction and therefore correspond by marker identity to the vacuum
  simulation markers. The workflow compares like-named modes; it does not
  resample the experimental trace at the same uncorrected numeric GHz value.
- Port extension is assumed already applied to the measurement files. The
  workflow must never apply it again. The numeric extension length is metadata
  provenance only.

## Implementation steps

1. Extend `deflector_tuning.tuning_campaign` with a small
   `SimulationReference` dataclass and role-keyed references. Keep the legacy
   single `campaign.sim` form readable so existing campaign files and tests do
   not break.
2. Add pure comparison helpers that:
   - read marker tables,
   - select configured positions and frequency markers,
   - compute wrapped `after - before` phase,
   - check the 180-degree anchor,
   - call the existing phase-radius-equivalence analysis.
3. Add a workflow that ensures missing baseline and simulation table products
   are analyzed, writes comparison tables/figure into the tuned result folder,
   and registers those files in the manifest.
4. Call that workflow only after `_tune_` metadata registration succeeds in
   `run_folder_analysis.py`.
5. Verify with focused unit tests, then run the actual 13.5 N m tuned dataset
   and visually inspect the generated figure.

## Expected outputs

- `tables/tuning_<family>_phase_change.csv`
- `tables/tuning_<family>_radius_equivalence.csv`
- `tables/tuning_<family>_anchor_check.csv`
- `figures/tuning_comparison/<family>_phase_radius_equivalence.png`
- manifest entries under `outputs.tuning_comparison`

If a tuned dataset contains only iris or only cell measurements, only that
family is produced. A `_celliris_` dataset may produce both when both matching
simulation references and configured positions are present.
