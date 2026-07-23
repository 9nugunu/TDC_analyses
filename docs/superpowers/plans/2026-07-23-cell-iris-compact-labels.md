# Cell-Iris Compact Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shorten all visible text in the `cell_iris_response` figures without changing their data or filenames.

**Architecture:** Keep the existing plotting entry point and replace only the display strings passed into the shared plotting helpers.  The x-tick formatter remains the single source for transition labels.

**Tech Stack:** Python, pandas, Matplotlib, pytest.

## Global Constraints

- Preserve all calculations, table contracts, plot order, and output filenames.
- Use compact scientific symbols exactly as approved in the design.

---

### Task 1: Compact plot copy

**Files:**
- Modify: `deflector_tuning/visualization/cell_iris_response_plots.py`
- Test: `tests/test_cell_iris_response_plots.py`

**Interfaces:**
- Consumes: `plot_cell_iris_response_comparison(comparison, output_dir, config=None)`.
- Produces: The same four `OrderedDict` keys and paths with compact visible labels.

- [x] **Step 1: Write the failing test**

```python
assert axis_calls == [
    ("Transition", "Iris / Cell", "Iris/Cell |Y11| ratio"),
    ("Transition", r"$|\Delta\phi_I| / |\Delta\phi_C|$", r"Iris/Cell $\Delta\phi$ ratio"),
    ("Transition", r"$|\phi - \phi_0|$ [deg]", "Phase residual"),
    ("Transition", r"$|\phi - \phi_{axis}|$ [deg]", "Axis error"),
]
```

- [x] **Step 2: Run test to verify it fails**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests\test_cell_iris_response_plots.py::test_plot_cell_iris_response_comparison_uses_compact_visible_labels`

Expected: FAIL because the current labels are verbose.

- [x] **Step 3: Write minimal implementation**

```python
title="Iris/Cell |Y11| ratio"
ylabel="Iris / Cell"
ax.axhline(..., label="equal")
```

Apply the approved copy to all four calls and compact `_comparison_label`.

- [x] **Step 4: Run tests and regenerate**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests\test_cell_iris_response_plots.py`

Then run the existing s004 folder analysis to regenerate the four figures.

- [x] **Step 5: Verify the rendered figure**

Inspect `fig/analyses/raw_sweep_260722_tune_s004/figures/cell_iris_response/`.
