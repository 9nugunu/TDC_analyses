# Y11 Admittance Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load CST `.y1p` sweeps and render all Y11 sweep views directly from the CST-exported raw admittance values.

**Architecture:** A Y11 folder is detected before the ordinary S-parameter loader. Direct CST Y11 values and Result Navigator geometry metadata are loaded, marker frequencies are sampled, and raw Y11 frequency, complex-plane, polar, and `r_c` grid views are rendered. No S11, Gamma, 50 ohm renormalization, or normalized admittance transform is applied.

**Tech Stack:** Python, pandas, matplotlib, pytest.

## Global Constraints

- Preserve CST-exported Y values and record the Touchstone `R` field as metadata.
- Keep existing S-parameter behavior unchanged for S-parameter datasets.

---

### Task 1: Parse direct CST Y Touchstone files

**Files:**

- Modify: `deflector_tuning/data_loading/readers/touchstone_reader.py`
- Create: `deflector_tuning/data_loading/admittance.py`
- Test: `tests/test_y_admittance_workflow.py`

- [x] Write a failing test for `.y1p` RI parsing and a two-file Y11 table with Result Navigator `r_c` metadata.
- [x] Verify the test fails because the current Touchstone reader only accepts `.sNp` extensions.
- [x] Accept `.yNp` extensions in the generic parser and add a Y11-only table loader that rejects non-Y headers and non-one-port files.
- [x] Verify the loader test passes.

### Task 2: Render raw Y11 views

**Files:**

- Modify: `deflector_tuning/data_loading/admittance.py`
- Test: `tests/test_y_admittance_workflow.py`

- [x] Add nearest-frequency sampling that retains `f_2pi3`, `f_mean`, and `f_pi2` labels.
- [x] Render raw Re(Y11)/Im(Y11) frequency traces and marker points.
- [x] Render raw complex-Y11 and raw Y11 polar views.
- [x] Render raw Y11 marker sweeps against `r_c`.

### Task 3: Route `.y1p` folders through the raw-Y runner lane

**Files:**

- Modify: `deflector_tuning/runner.py`
- Test: `tests/test_y_admittance_workflow.py`

- [x] Write a runner test that invokes `run_folder_analysis` for a `.y1p` folder and expects raw Y renderers.
- [x] Detect a Y Touchstone folder, write raw Y tables, and retain direct Y columns in marker points.
- [x] Verify the test passes and run the full regression suite.
