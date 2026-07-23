# Nodal Transition Radius Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plot the `f_2pi3` cell-to-cell and iris-to-iris reflection-phase advances versus regular-cell radius for the fine nodal-shift sweep.

**Architecture:** Add one explicit CLI script that consumes the reference and fine-sweep `marker_pts.csv` tables. The script computes same-family advances from fixed upstream reference positions to swept downstream positions, writes a provenance-bearing CSV, and renders a two-color plot with the 240-degree target and zero-offset annotations.

**Tech Stack:** Python, pandas, matplotlib, pytest

## Global Constraints

- Use `sim_sweep_260618_Nodalshift` only for the fixed `3.5` cell and `4.0` iris references.
- Use `sim_sweep_260619_Nodalshift_Fine` only for the swept `4.5` cell and `5.0` iris points.
- Include only `f_2pi3`; do not mix `f_mean` or `f_pi2`.
- Use forward-wrapped reflection phase advance in `[0, 360)` with an ideal target of 240 degrees.
- Do not use `sim_sweep_260620_FullstructureSweep_ports_swapped` as a numerical anchor because its geometry contract differs.

---

### Task 1: Compute and plot the transition sweep

**Files:**
- Create: `scripts/plot_nodal_transition_radius_sweep.py`
- Test: `tests/test_plot_nodal_transition_radius_sweep.py`

**Interfaces:**
- Consumes: two marker-point DataFrames containing `marker_name`, `tune_position`, `s_phase_deg`, and the fine table column `sim_offset_cell_03`
- Produces: `build_transition_sweep(...) -> pandas.DataFrame` and `plot_transition_sweep(...) -> pathlib.Path`

- [ ] **Step 1: Write failing tests**

Test that cell `3.5->4.5` and iris `4.0->5.0` advances are forward-wrapped, that only `f_2pi3` is used, and that the plot contains distinct family colors, a 240-degree guide, and zero-offset annotations.

- [ ] **Step 2: Run tests and verify expected failure**

Run: `conda run -n sys_env1 python -m pytest -q tests/test_plot_nodal_transition_radius_sweep.py`

Expected: collection failure because the script does not exist.

- [ ] **Step 3: Implement the minimal script**

Validate the required columns and unique reference rows, build one output row per fine-sweep offset, calculate `phase_advance_deg = (phase_to_deg - phase_from_deg) % 360`, and render the two series.

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
conda run -n sys_env1 python -m pytest -q tests/test_plot_nodal_transition_radius_sweep.py
conda run -n sys_env1 python -m pytest -q
```

Expected: all tests pass.

### Task 2: Generate and verify the production artifact

**Files:**
- Generate: `fig/analyses/sim_sweep_260619_Nodalshift_Fine/tables/f_2pi3_same_family_transition_vs_radius.csv`
- Generate: `fig/analyses/sim_sweep_260619_Nodalshift_Fine/figures/nodal_shift/f_2pi3_same_family_transition_vs_radius.png`

**Interfaces:**
- Consumes: the two production `marker_pts.csv` tables
- Produces: the exact CSV and PNG requested for review

- [ ] **Step 1: Run the script with explicit source paths**

Use a base radius of `57.09 mm` and the production table paths.

- [ ] **Step 2: Verify numerically and visually**

Confirm 21 radius offsets, both families at each offset, a 240-degree target line, distinct colors, and readable zero-offset values.
