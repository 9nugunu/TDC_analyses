# Concise `r_c` Phase-State Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace verbose and inconsistent `r_c` phase-map state wording with concise, consistent labels and a clearer artifact name.

**Architecture:** Keep the existing phase-line plotting and radius calculations unchanged. Update the centralized `plot_phase_rc_map()` label defaults, add a thin four-state wrapper for the auxiliary `f_mean = 0°` guide, and regenerate the requested PNG.

**Tech Stack:** Python, pandas, Matplotlib, pytest.

## Global Constraints

- Do not change simulated data, fitted radii, guide positions, or phase calculations.
- Keep labels explicit: `Before tuning`, `Torque 13.5`, `Design`, and `f_mean = 0°`.
- Do not commit or alter unrelated existing worktree changes.

---

### Task 1: Update centralized phase-map labels

**Files:**
- Modify: `deflector_tuning/visualization/grid_scan_phase_line_plots.py`
- Test: `tests/test_grid_scan_phase_line_plots.py`

**Interfaces:**
- `plot_phase_rc_map(line_scan, rc_fit, output_path)` keeps its existing signature and output behavior.
- The helper continues to pass `candidate_label`, `before_label`, and `target_label` to the renderer, but with concise state names.

- [x] Write the failing assertion expecting `Torque 13.5`, `Before tuning`, and `Design` labels.
- [x] Run the focused test and confirm it fails on the old label text.
- [x] Replace the three label format strings with the approved concise wording.
- [x] Run the targeted test and the complete `tests/test_grid_scan_phase_line_plots.py` module.

### Task 2: Add the four-state renderer

**Files:**
- Modify: `deflector_tuning/visualization/grid_scan_phase_line_plots.py`
- Create: `scripts/plot_rc_phase_states.py`
- Test: `tests/test_grid_scan_phase_line_plots.py`

**Interfaces:**
- `plot_phase_rc_states(line_scan, output_path, *, before_r_c_mm, current_r_c_mm, design_r_c_mm, f_mean_zero_r_c_mm, fixed_w_c=19.3224, config=None)` renders the existing phase curves with four named guides.
- The CLI accepts the same four radius values and writes the requested output path.

- [x] Assert the four approved labels and the design guide flag in the plotting test.
- [x] Run the new test and confirm it fails before the wrapper exists.
- [x] Add the wrapper and CLI with no changes to the phase data or radius calculations.
- [x] Run the focused plotting test module and confirm all tests pass.

### Task 3: Regenerate the renamed artifact

**Files:**
- Create: `fig/analyses/sim_grid_260701_1DRcFine/figures/grid_scan_sparameter_phase_r_c_line_scan/rc_phase_states.png`

**Interfaces:**
- Consume the existing Fine `r_c` line-scan CSV and preserve the approved before/current/design radius values.
- Produce a non-empty PNG with the concise state labels.

- [x] Run `scripts/plot_rc_phase_states.py` with the Fine line-scan CSV and approved state values.
- [x] Confirm `rc_phase_states.png` exists and has non-zero size.
