# Position Phase Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a tested, reproducible grouped bar chart comparing wrapped S11 phase at positions 1.0 and 2.0 across the three operating frequency markers.

**Architecture:** Add one focused visualization module that validates and reshapes the existing marker-point table, then renders the two-level grouped bars with shared project styling. Add a thin CLI that loads a table and passes explicit positions and an output path to that module.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, pytest

## Global Constraints

- Preserve the signed wrapped phase in `s_phase_deg`.
- Do not add a dataset-specific conditional.
- Do not modify raw or preprocessed data.
- Do not commit, push, merge, reset, or delete files.

---

### Task 1: Chart-ready comparison table and grouped bar renderer

**Files:**
- Create: `deflector_tuning/visualization/position_phase_bar_plots.py`
- Create: `tests/test_position_phase_bar_plots.py`

**Interfaces:**
- Consumes: `pandas.DataFrame` marker points and `positions: Sequence[float]`
- Produces: `build_position_phase_bar_table(...) -> pandas.DataFrame`
- Produces: `plot_position_phase_bars(...) -> pathlib.Path`

- [ ] **Step 1: Write failing table-contract tests**

  Test that the builder returns six rows ordered by marker then position, keeps
  the six signed phase values unchanged, and raises a clear `ValueError` for a
  missing or duplicate marker-position pair.

- [ ] **Step 2: Run the focused test and verify RED**

  Run: `pytest tests/test_position_phase_bar_plots.py -q`

  Expected: collection failure because
  `deflector_tuning.visualization.position_phase_bar_plots` does not exist.

- [ ] **Step 3: Implement the minimal table builder**

  Validate required columns, filter the explicit positions and marker order,
  verify one row per pair, and return the stable chart-ready columns without
  changing `s_phase_deg`.

- [ ] **Step 4: Run the table tests and verify GREEN**

  Run: `pytest tests/test_position_phase_bar_plots.py -q`

  Expected: all current table-contract tests pass.

- [ ] **Step 5: Write a failing rendering test**

  Assert that `plot_position_phase_bars` writes the requested PNG and that the
  Matplotlib figure is closed afterward.

- [ ] **Step 6: Run the rendering test and verify RED**

  Run: `pytest tests/test_position_phase_bar_plots.py -q`

  Expected: failure because the renderer is missing.

- [ ] **Step 7: Implement the minimal renderer**

  Draw three grouped mode pairs, two-level x labels, direct degree labels, a
  zero line, fixed signed-phase limits, explicit colors/hatch, neutral title,
  subtitle, and shared project typography.

- [ ] **Step 8: Run the focused tests and verify GREEN**

  Run: `pytest tests/test_position_phase_bar_plots.py -q`

  Expected: all focused tests pass.

### Task 2: CLI and requested artifact

**Files:**
- Create: `scripts/plot_position_phase_bars.py`
- Create: `tests/test_plot_position_phase_bars_cli.py`
- Generate: `fig/analyses/raw_sweep_260701_iris_portE/figures/phase_bar/position_1p0_vs_2p0_phase_bars.png`
- Generate: `fig/analyses/raw_sweep_260701_iris_portE/figures/phase_bar/position_1p0_vs_2p0_phase_bars.csv`

**Interfaces:**
- Consumes: `--marker-points`, `--positions`, and `--output`
- Produces: the requested PNG and its chart-ready CSV sidecar

- [ ] **Step 1: Write the failing CLI test**

  Invoke `main(...)` with a temporary marker table and assert that both PNG and
  CSV outputs exist with the expected six chart rows.

- [ ] **Step 2: Run the CLI test and verify RED**

  Run: `pytest tests/test_plot_position_phase_bars_cli.py -q`

  Expected: collection failure because the CLI module does not exist.

- [ ] **Step 3: Implement the thin CLI**

  Parse explicit paths and positions, load `marker_pts.csv`, build and save the
  chart-ready table, and call the tested renderer.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run: `pytest tests/test_position_phase_bar_plots.py tests/test_plot_position_phase_bars_cli.py -q`

  Expected: all focused tests pass.

- [ ] **Step 5: Generate the requested artifact**

  Run:

  `python scripts/plot_position_phase_bars.py --marker-points fig/analyses/raw_sweep_260701_iris_portE/tables/marker_pts.csv --positions 1.0 2.0 --output fig/analyses/raw_sweep_260701_iris_portE/figures/phase_bar/position_1p0_vs_2p0_phase_bars.png`

- [ ] **Step 6: Verify numerically and visually**

  Compare the six CSV values with `marker_pts.csv`, inspect the final PNG, and
  run the focused tests once more before reporting completion.
