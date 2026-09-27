# Geometry Sweep Axis and Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit geometry sweep axis and reference value to the standard one-folder analysis, then generate and interpret the z-length sweep outputs.

**Architecture:** Optional CLI arguments propagate through the runner into the existing marker pipeline. `compute_geometry_phase_response()` remains the single owner of axis validation, Cell/Iris pairing, and reference subtraction; the manifest records the resolved request.

**Tech Stack:** Python 3.11, pandas, NumPy, matplotlib, pytest

## Global Constraints

- Do not introduce a dataset-name branch.
- Preserve `data/sim` routing and original Touchstone files.
- Treat `sim_L_c` as geometry and `NumDepth`/`DepthPlunger` as observation-state metadata.
- Do not call the Cell 1.5 to Iris 2.0 difference a periodic phase advance.
- Preserve unrelated working-tree changes.
- Do not commit, push, merge, reset, or delete files.

---

### Task 1: Geometry response API

**Files:**
- Modify: `tests/test_geometry_phase_response.py`
- Modify: `deflector_tuning/analysis/geometry_phase_response.py`

**Interfaces:**
- Consumes: marker-point `DataFrame`, optional `sweep_axis: str | None`, optional `sweep_base: float | None`
- Produces: `compute_geometry_phase_response(..., sweep_axis=..., sweep_base=...) -> DataFrame`

- [ ] Add a failing test with varying `sim_L_c` and state-dependent `sim_DepthPlunger` that requests `sim_L_c` and 29.148.
- [ ] Run the focused test and confirm it fails because the function lacks the explicit keyword arguments.
- [ ] Add keyword-only parameters, validate the axis/base, and use the requested baseline.
- [ ] Add failure tests for missing/constant axes and an unsampled base.
- [ ] Run `tests/test_geometry_phase_response.py`.

### Task 2: Pipeline, CLI, and manifest propagation

**Files:**
- Modify: `tests/test_marker_analysis_pipeline.py`
- Modify: `tests/test_run_folder_analysis_cli.py`
- Modify: `tests/test_runner.py`
- Modify: `deflector_tuning/analysis/marker_pipeline.py`
- Modify: `deflector_tuning/runner.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `run_folder_analysis.py`

**Interfaces:**
- Consumes: `geometry_sweep_axis: str | None`, `geometry_sweep_base: float | None`
- Produces: CLI-to-analysis propagation and manifest `geometry_phase_response` metadata

- [ ] Add failing pipeline and CLI propagation tests.
- [ ] Add a failing runner test that checks the manifest metadata.
- [ ] Run the focused tests and confirm the missing propagation failures.
- [ ] Thread the optional values through each layer and write them into both full and tables-only manifests.
- [ ] Run the focused pipeline, CLI, runner, and manifest tests.

### Task 3: Real dataset generation and verification

**Files:**
- Generate: `fig/analyses/sim_sweep_260728_zlen/tables/*.csv`
- Generate: `fig/analyses/sim_sweep_260728_zlen/figures/**/*.png`
- Generate: `fig/analyses/sim_sweep_260728_zlen/manifest.json`

**Interfaces:**
- Consumes: `data/sim/sim_sweep_260728_zlen`, the default simulation dispersion dataset, axis `sim_L_c`, base `29.148`
- Produces: reproducible tables, figures, and manifest

- [ ] Run the one-folder CLI with four file workers and the explicit geometry options.
- [ ] Verify 22 source files, 66 marker rows, 33 geometry-response rows, and `sweep_base=29.148`.
- [ ] Open the combined absolute-phase and phase-pickup figures and inspect labels, marker order, continuity, and reference zeros.
- [ ] Run the complete focused test set and `git diff --check`.
- [ ] Interpret the three marker sensitivities and clearly separate fixed-marker phase response from eigenfrequency sensitivity.
