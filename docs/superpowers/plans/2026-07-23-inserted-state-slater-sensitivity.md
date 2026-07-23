# Inserted-State Slater Sensitivity Implementation Plan

> Approved design: `docs/superpowers/specs/2026-07-23-inserted-state-slater-sensitivity-design.md`
>
> Repository constraint: preserve the user's existing uncommitted work and do
> not commit, branch, push, reset, or delete files.

## Task 1: Protect the inserted-state physics contract

**Files**

- Modify: `tests/test_field3d_analysis.py`
- Verify: `tests/test_field3d_analysis.py`

1. Extend the synthetic field writer so individual axial planes can have
   distinct amplitudes and different axial coverage.
2. Add a test whose NoPlunger and inserted fields deliberately differ.
3. Assert that the inserted calculation samples `z_tip - dz`, integrates the
   inserted field in the next vacuum slab, and normalizes by that inserted
   case's own export-region energy.
4. Run:

   ```powershell
   C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_analysis.py::test_slater_table_adds_inserted_state_vacuum_side_term
   ```

   Confirm the test fails because the inserted-state columns do not yet exist.

## Task 2: Implement the inserted-state local term

**Files**

- Modify: `deflector_tuning/analysis/field3d.py`
- Modify: `tests/test_field3d_analysis.py`

1. Select one canonical inserted case for every reference position, preferring
   the case with the largest E/H row coverage and then a named NumDepth case.
2. Locate the grid plane immediately before the detected PEC tip.
3. Load the inserted E/H fields on that plane and integrate the circular
   cross-section with that case's own voxel spacing.
4. Add the approved inserted-state columns while leaving all existing baseline
   columns numerically unchanged.
5. Add tests for duplicate-position selection and the missing preceding-vacuum
   plane error.
6. Run:

   ```powershell
   C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_analysis.py
   ```

## Task 3: Expose both field bases in the workflow

**Files**

- Modify: `deflector_tuning/workflows/field3d.py`
- Modify: `tests/test_field3d_workflow.py`

1. Add a failing workflow assertion for the new CSV columns, schema version,
   and manifest definitions of the NoPlunger and inserted-state quantities.
2. Bump the field3d cache schema so old outputs cannot be reused.
3. Record the common baseline, per-position inserted basis, sampling side,
   normalization basis, fixed-frequency limitation, and absence of an absolute
   tuner-coefficient claim.
4. Run:

   ```powershell
   C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_workflow.py
   ```

## Task 4: Compare the signed local terms in the figure

**Files**

- Modify: `deflector_tuning/visualization/field3d_plots.py`
- Modify: `tests/test_field3d_plots.py`

1. Add a failing plot assertion for separate NoPlunger-baseline and
   inserted-state signed `K = U_E - U_H` series.
2. Retain the Cell/Iris grouping and existing E/H bars, then add a clearly
   labelled inserted-state line without presenting either series as a measured
   frequency shift.
3. Run:

   ```powershell
   C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_plots.py
   ```

## Task 5: Verify code and regenerate the real analysis

**Files**

- Verify: `tests/test_field3d_analysis.py`
- Verify: `tests/test_field3d_workflow.py`
- Verify: `tests/test_field3d_plots.py`
- Regenerate:
  `fig/analyses/sim_profile_260723_3DEMfield/tables/slater_pos.csv`
- Regenerate:
  `fig/analyses/sim_profile_260723_3DEMfield/figures/field3d/slater_pos.png`
- Regenerate:
  `fig/analyses/sim_profile_260723_3DEMfield/manifest.json`

1. Run the focused field3d test suite:

   ```powershell
   C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_field3d_loading.py tests/test_field3d_analysis.py tests/test_field3d_plots.py tests/test_field3d_workflow.py
   ```

2. Run the existing CLI/runner path on
   `data/sim/sim_profile_260723_3DEMfield` so the normal cache and manifest
   routing are exercised.
3. Inspect the regenerated CSV and manifest. Confirm that the four physical
   positions use full NumDepth cases and that every inserted sample lies one
   grid step before its detected tip.
4. Visually inspect `slater_pos.png`.
5. Run the full test suite and report the exact command and result.

