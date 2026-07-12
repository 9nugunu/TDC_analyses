# Y11 Admittance Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load CST `.y1p` sweeps without treating them as S11 and render their three marker-frequency complex-Y11 trajectories.

**Architecture:** A narrow Y-only workflow detects `.y1p` files before the standard S-parameter runner. It reads direct CST Y11 values, joins CST Result Navigator geometry metadata, samples the existing dispersion markers, and writes a Cartesian complex-admittance figure. The S-parameter loader and KYHL transform remain unchanged.

**Tech Stack:** Python, pandas, matplotlib, pytest.

## Global Constraints

- Do not reinterpret Y11 as S11 or pass it through S11 dB/phase plots.
- Preserve CST-exported Y values and record the Touchstone `R` field as metadata.
- Keep existing S-parameter behavior unchanged.

---

### Task 1: Parse direct CST Y Touchstone files

**Files:**

- Modify: `deflector_tuning/data_loading/readers/touchstone_reader.py`
- Create: `deflector_tuning/data_loading/admittance.py`
- Test: `tests/test_y_admittance_workflow.py`

- [ ] Write a failing test for `.y1p` RI parsing and a two-file Y11 table with Result Navigator `r_c` metadata.
- [ ] Verify the test fails because the current Touchstone reader only accepts `.sNp` extensions.
- [ ] Accept `.yNp` extensions in the generic parser and add a Y11-only table loader that rejects non-Y headers and non-one-port files.
- [ ] Verify the loader test passes.

### Task 2: Sample marker frequencies and render complex Y11

**Files:**

- Modify: `deflector_tuning/data_loading/admittance.py`
- Create: `deflector_tuning/visualization/admittance_sweep_plots.py`
- Test: `tests/test_y_admittance_workflow.py`

- [ ] Write a failing test asserting one nearest Y11 point per marker and per Result Navigator run, plus an output figure.
- [ ] Add nearest-frequency sampling that retains `f_2pi3`, `f_mean`, and `f_pi2` labels.
- [ ] Draw Re(Y11) versus Im(Y11) in mS, with an independent trajectory per marker and start/end sweep markers.
- [ ] Verify the test passes.

### Task 3: Route `.y1p` folders through the Y-only runner lane

**Files:**

- Modify: `deflector_tuning/runner.py`
- Test: `tests/test_y_admittance_workflow.py`

- [ ] Write a failing runner test that invokes `run_folder_analysis` for a `.y1p` folder and expects only Y11 outputs.
- [ ] Detect a Y Touchstone folder before S11 marker analysis, write `markers.csv` and `y11_marker_points.csv`, and render `figures/y11_complex/y11_marker_sweep.png`.
- [ ] Verify the test passes and run the focused regression suite.
