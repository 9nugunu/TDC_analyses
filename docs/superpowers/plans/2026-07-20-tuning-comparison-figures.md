# Tuning Comparison Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` for implementation and `superpowers:verification-before-completion` before claiming completion.

**Goal:** Automatically map a tuned experimental dataset onto the configured 1D `r_c` simulation and emit two concise comparison figures plus three concise evidence tables.

**Architecture:** Keep campaign routing in `tuning_campaign.py`, numerical calibration in the phase/radius analysis module, plotting in visualization modules, and orchestration in the tuning comparison workflow. Experimental marker phases remain temperature-corrected measurements and are paired with vacuum-simulation markers by marker identity. Port extension is checked, never reapplied.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, pytest, YAML campaign configuration.

## Global Constraints

- Do not add dataset-name-specific branches.
- Trigger tuning comparison from `_tune_` dataset identity and campaign role references.
- Use only configured 1D `r_c` simulation data for this comparison.
- Do not interpret equivalent `r_c` changes as literal bolt travel.
- Do not commit or push changes.

---

## Task 1: Protect the radius-state calibration

**Files:**
- Modify: `tests/test_tuning_simulation_comparison.py`
- Modify: `deflector_tuning/analysis/phase_radius_equivalence.py`

- [ ] Add a synthetic monotonic-line test for `fit_rc_states()`.
- [ ] Verify the test fails before implementation.
- [ ] Implement current-state inversion from the measured `f_mean` phase relative to the design simulation phase.
- [ ] Implement baseline-state fitting from all three measured phase changes while holding the current state fixed.
- [ ] Return explicit baseline, current, and design rows with fit diagnostics.
- [ ] Run the focused calibration test.

## Task 2: Introduce concise plotting functions

**Files:**
- Modify: `tests/test_phase_radius_equivalence.py`
- Modify: `tests/test_tuning_phase_comparison_plots.py`
- Modify: `deflector_tuning/visualization/grid_scan_phase_line_plots.py`
- Modify: `deflector_tuning/visualization/phase_radius_equivalence_plots.py`
- Modify: `scripts/plot_rc_candidate_on_phase_line.py`
- Modify: `scripts/plot_experiment_simulation_phase_bars.py`

- [ ] Update tests to require `plot_phase_rc_map()` and `plot_phase_cmp_bars()`.
- [ ] Verify the renamed-function tests fail.
- [ ] Replace the verbose phase/radius plot entry point with `plot_phase_rc_map()` using baseline, current, and design vertical lines.
- [ ] Replace the verbose bar-plot entry point with `plot_phase_cmp_bars()` and a generic current-state label.
- [ ] Update script callers without compatibility aliases.
- [ ] Run the focused plotting tests.

## Task 3: Replace the automatic output contract

**Files:**
- Modify: `tests/test_tuning_simulation_comparison.py`
- Modify: `tests/test_workflow_manifest.py`
- Modify: `deflector_tuning/workflows/tuning_simulation_comparison.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `run_folder_analysis.py`
- Modify: `run_all_folder_analyses.py`

- [ ] Add tests for `tables/rc_fit.csv`, `tables/phase_cmp.csv`, and `tables/anchor_chk.csv`.
- [ ] Add tests for `figures/tuning_cmp/phase_rc_map.png` and `figures/tuning_cmp/phase_cmp_bars.png`.
- [ ] Add a manifest test for `outputs.tuning_cmp`.
- [ ] Verify the new workflow tests fail before implementation.
- [ ] Implement `run_tuning_cmp()` and `register_tuning_cmp()` using the concise output names.
- [ ] Remove automatic generation and registration of the old radius-equivalence figure and verbose tables.
- [ ] Run the focused workflow and CLI tests.

## Task 4: Generate and inspect the real outputs

**Files:**
- Generate under: `fig/analyses/raw_sweep_260701_iris_tune_Torque13p5/`

- [ ] Run folder analysis for the tuned experimental dataset.
- [ ] Inspect the three generated CSV files for marker identity, fit values, and anchor status.
- [ ] Visually inspect both generated figures.
- [ ] Confirm the result is described as an equivalent simulation coordinate, not a mechanical displacement.

## Task 5: Remove only verified legacy artifacts

**Files:**
- Delete known obsolete generated figures and their replaced verbose comparison tables only after Task 4 succeeds.

- [ ] Resolve every deletion target and verify that it is under the workspace `fig/analyses` directory.
- [ ] Delete the old `phase_rc_experiment_calibrated_to_design.png`.
- [ ] Delete the old `experiment_vs_simulation_phase_bars.png` and its replaced comparison CSV.
- [ ] Delete the superseded automatic `iris_phase_radius_equivalence.png` and verbose tuning tables.
- [ ] Remove only empty legacy output directories; preserve shared simulation line-scan CSV files.

## Task 6: Final verification

- [ ] Run the focused test suite covering campaign routing, fitting, plotting, workflow, manifests, and CLI entry points.
- [ ] Run `git diff --check`.
- [ ] Confirm the two new figures and three new tables exist and the named obsolete artifacts do not.
- [ ] Report the exact commands and real results without committing changes.
