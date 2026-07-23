# Automatic 3D Field Profile Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run reusable 3D complex E/H field and Slater-position analysis by passing only a CST profile dataset folder to the existing runner.

**Architecture:** Add a header-only 3D detector and paired chunked loader, a numerical analysis module, compact plots, and a profile-only workflow with manifest fingerprint reuse. Route this workflow before the existing 1D profile loader so large 3D files are never read by `Path.read_text`.

**Tech Stack:** Python 3.11, NumPy, pandas, Matplotlib, pytest.

## Global Constraints

- Preserve centralized `data/sim`, `data/raw`, and `data/prepro` routing.
- Do not add dataset-name-specific branches.
- Keep existing 1D profile behavior unchanged.
- Do not require YAML or new CLI options.
- Do not claim an absolute Slater frequency shift.
- Do not commit because the user requested execution and saved results, not a Git commit.

---

### Task 1: Detect and stream paired 3D field exports

**Files:**
- Create: `deflector_tuning/data_loading/field3d.py`
- Test: `tests/test_field3d_loading.py`

**Interfaces:**
- Produces: `Field3DPair`, `Field3DFileSummary`, `is_field3d_export(path)`, `find_field3d_pairs(path)`, and `summarize_field3d_file(path, chunk_rows=...)`.

- [ ] **Step 1: Write failing detector and pairing tests**

Create tiny CST-style E/H files and assert header-only detection, `default`
case naming, suffix case naming, and E/H pairing.

- [ ] **Step 2: Run the tests and verify the imports fail**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_loading.py`

Expected: FAIL because `deflector_tuning.data_loading.field3d` does not exist.

- [ ] **Step 3: Implement the minimal detector, pairing, and bounded chunk summary**

Use the first header line for detection. Read numeric rows in bounded NumPy
chunks, accumulate grid axes, component magnitude-squared sums, and closest
axis energy by \(z\).

- [ ] **Step 4: Run the loading tests**

Run the command from Step 2.

Expected: PASS.

### Task 2: Infer geometry and calculate local Slater terms

**Files:**
- Create: `deflector_tuning/analysis/field3d.py`
- Test: `tests/test_field3d_analysis.py`

**Interfaces:**
- Consumes: `Field3DPair`, `Field3DFileSummary`.
- Produces: `combine_field3d_summaries(...)`, `detect_plunger_tip_mm(...)`,
  `infer_plunger_radius_mm(...)`, and `compute_slater_position_table(...)`.

- [ ] **Step 1: Write failing synthetic geometry tests**

Use a small rectangular grid with a central zero cylinder and constant complex
fields. Assert the detected tip, inferred equivalent radius, component energy,
signed \(K_{E-H}\), and normalized magnitude.

- [ ] **Step 2: Run and verify expected failures**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_analysis.py`

Expected: FAIL because the analysis functions do not exist.

- [ ] **Step 3: Implement geometry inference and disk integration**

Infer the axis-connected zero component with four-neighbor connectivity.
Sample no-plunger E/H on the nearest reference plane and integrate one
effective grid slab.

- [ ] **Step 4: Run the analysis tests**

Run the command from Step 2.

Expected: PASS.

### Task 3: Save compact tables, figures, and reusable manifest

**Files:**
- Create: `deflector_tuning/visualization/field3d_plots.py`
- Create: `deflector_tuning/workflows/field3d.py`
- Test: `tests/test_field3d_workflow.py`

**Interfaces:**
- Consumes: discovered field pairs and analysis tables.
- Produces: `run_field3d_analysis(...) -> RunResult` and
  `load_cached_field3d_result(...) -> RunResult | None`.

- [ ] **Step 1: Write a failing end-to-end workflow test**

Assert exact table names, figure names, manifest mode `field3d`, and that a
second unchanged run returns cached products without invoking the analyzer.

- [ ] **Step 2: Run and verify expected failure**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_workflow.py`

Expected: FAIL because the workflow does not exist.

- [ ] **Step 3: Implement tables, concise figures, manifest metadata, and cache**

Reuse `PlotConfig`, `apply_axis_text_style`, and `save_figure`. Store source
signatures and schema version under `field3d` in `manifest.json`.

- [ ] **Step 4: Run the workflow test**

Run the command from Step 2.

Expected: PASS.

### Task 4: Route 3D profiles from the existing folder runner

**Files:**
- Modify: `deflector_tuning/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `find_field3d_pairs` and `run_field3d_analysis`.
- Produces: unchanged public `run_folder_analysis(...) -> RunResult`.

- [ ] **Step 1: Write a failing runner-routing test**

Create a profile-category folder with a tiny paired 3D export and assert that
the result mode is `("field3d",)` and the 1D loader is not called.

- [ ] **Step 2: Run and verify expected failure**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_runner.py -k field3d`

Expected: FAIL because 3D routing is absent.

- [ ] **Step 3: Add 3D header routing before 1D profile discovery**

Keep public seams patchable from `deflector_tuning.runner` and pass the
existing output, table, figure, and manifest locations through unchanged.

- [ ] **Step 4: Run focused and profile regression tests**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_runner.py -k "field3d or profile"`

Expected: PASS.

### Task 5: Execute and verify the real dataset

**Files:**
- Generate: `fig/analyses/sim_profile_260723_3DEMfield/`

**Interfaces:**
- Consumes: `data/sim/sim_profile_260723_3DEMfield`.
- Produces: the three CSV tables, two PNG figures, and manifest.

- [ ] **Step 1: Run the focused test suite**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_loading.py tests/test_field3d_analysis.py tests/test_field3d_workflow.py tests/test_runner.py -k "field3d or profile"`

Expected: PASS.

- [ ] **Step 2: Run the real folder**

Run:
`C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe run_folder_analysis.py sim_profile_260723_3DEMfield`

Expected: successful `field3d` analysis with printed output paths.

- [ ] **Step 3: Inspect numerical and visual outputs**

Read all three CSV files, confirm inferred tip order and radius, then open both
PNG files and verify labels, units, legend, and visible bars.

- [ ] **Step 4: Verify automatic reuse**

Run the same folder command again and confirm the log reports reuse without
streaming the source files.

