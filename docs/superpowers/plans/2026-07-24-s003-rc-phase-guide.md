# S003 r_c Phase Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the measured S003 tuning state as an explicitly labeled vertical guide to the existing simulated r_c phase-line figure.

**Architecture:** Extend the existing `plot_phase_rc_states()` wrapper with one optional S003 radius and label. Keep the simulation line scan and all existing guides unchanged; obtain the S003 value from its generated `rc_fit.csv` and regenerate a separate PNG artifact.

**Tech Stack:** Python, pandas, matplotlib, pytest, uv.

## Global Constraints

- Preserve the existing simulation/data routing and phase definitions.
- Treat every `r_c` as a simulation-equivalent coordinate, not physical bolt travel.
- Do not overwrite the existing four-guide artifact; write a new S003-inclusive PNG.

---

### Task 1: Add the S003 guide to the reusable plot wrapper

**Files:**
- Modify: `deflector_tuning/visualization/grid_scan_phase_line_plots.py`
- Test: `tests/test_grid_scan_phase_line_plots.py`

**Interfaces:**
- Add `s003_r_c_mm: float` to `plot_phase_rc_states()`.
- Render it as `S003 tuning: r_c = <value> mm` using the existing vertical-guide renderer.

- [x] Add a failing assertion that the wrapper passes the S003 radius and label to the renderer.
- [x] Run the focused test and confirm it fails before implementation.
- [x] Implement the smallest wrapper/renderer change.
- [x] Run the focused test and confirm it passes.

### Task 2: Regenerate and verify the S003-inclusive artifact

**Files:**
- Modify: `scripts/plot_rc_phase_states.py`
- Create: `fig/analyses/sim_grid_260701_1DRcFine/figures/grid_scan_sparameter_phase_r_c_line_scan/rc_phase_states_s003.png`

**Interfaces:**
- Add the CLI option `--s003-r-c-mm`.
- Use s002 `r_c = 56.46085702115745 mm` from `fig/analyses/raw_sweep_260721_tune_s002_Torque13p5/tables/iris_rc_fit.csv` and S003 `r_c = 56.469945087687854 mm` from `fig/analyses/raw_sweep_260721_tune_s003_broken/tables/rc_fit.csv`.

- [x] Generate the new PNG with the Fine line-scan CSV and existing guide values.
- [x] Verify the output exists and the S003 guide is at `56.469945087687854 mm`.
- [x] Visually inspect the rendered PNG for the new label, units, and unclipped legend.
- [x] Run the focused test suite and report the result.
