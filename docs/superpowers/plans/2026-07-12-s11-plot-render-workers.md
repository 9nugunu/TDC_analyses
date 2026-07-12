# S11 Plot Render Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow a single dataset run to render independent S11 figure plans in separate processes without introducing nested multiprocessing in batch dataset runs.

**Architecture:** `build_s11_plot_plans` remains serial and continues to decide output names, titles, and data slices. A new renderer helper dispatches complete `S11PlotPlan` objects to top-level process workers only when the caller requests more than one plot worker; each worker owns its Matplotlib state and writes a unique PNG. The one-folder CLI exposes `--plot-workers`, while the multi-dataset runner keeps dataset-level multiprocessing and explicitly uses serial S11 rendering inside each dataset task.

**Tech Stack:** Python `concurrent.futures.ProcessPoolExecutor`, Matplotlib Agg backend, pandas, pytest.

## Global Constraints

- Preserve existing S11 filenames, titles, output paths, and returned `OrderedDict` order.
- Do not parallelize Matplotlib with threads.
- Do not create nested process pools in `run_all_folder_analyses.py` dataset workers.
- Clamp effective plot workers to the number of plans and at least one worker.
- Fall back to serial rendering for one worker or one plan.
- Reject duplicate planned output paths before dispatching workers.

---

### Task 1: Define the plot-worker behavior with tests

**Files:**
- Modify: `tests/test_s11_frequency_plots.py`
- Modify: `tests/test_run_folder_analysis_cli.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- The S11 renderer accepts `render_workers: int = 1`.
- The one-folder CLI exposes `--plot-workers` and stores the normalized value on parsed arguments.
- `run_folder_analysis` accepts `plot_workers` and forwards it only to S11 rendering.

- [x] **Step 1: Write failing tests**

Add tests that assert:

```python
def test_plot_s11_with_markers_parallel_render_preserves_plan_order(tmp_path, monkeypatch):
    plans = [
        S11PlotPlan("first", "position", tmp_path / "first.png", "first", pd.DataFrame(), pd.DataFrame()),
        S11PlotPlan("second", "position", tmp_path / "second.png", "second", pd.DataFrame(), pd.DataFrame()),
    ]
    monkeypatch.setattr(s11_frequency_plots, "build_s11_plot_plans", lambda *args, **kwargs: plans)
    monkeypatch.setattr(s11_frequency_plots, "_plot_one", lambda *args, **kwargs: args[2])
    result = plot_s11_with_markers(_sparameter_table(), _marker_points(), tmp_path, render_workers=2)
    assert list(result) == ["first", "second"]
```

Add CLI assertions for `--plot-workers`, and a runner test that verifies the value is forwarded to `plot_s11_with_markers`.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
pytest tests/test_s11_frequency_plots.py tests/test_run_folder_analysis_cli.py tests/test_runner.py -q
```

Expected: failures because `S11PlotPlan`, `plot_s11_with_markers`, and the CLI/runner do not yet accept the new worker behavior.

### Task 2: Implement safe S11 plan-level multiprocessing

**Files:**
- Modify: `deflector_tuning/visualization/s11_frequency_plots.py`

**Interfaces:**
- Add `render_workers: int = 1` to `plot_s11_with_markers`.
- Add a top-level `_render_s11_plan(plan, config)` worker returning `(plan.key, rendered_path)`.
- Add a helper that validates unique output paths, clamps workers, dispatches with `ProcessPoolExecutor`, and restores plan order.

- [x] **Step 1: Implement the minimal renderer**

Use serial `_plot_one` for one worker or one plan. For multiple plans, dispatch only the top-level worker with `ProcessPoolExecutor(max_workers=min(max(int(render_workers), 1), len(plans)))`; collect futures by plan index and return an `OrderedDict` in original order. Raise `ValueError` when two plans have the same `output_path`.

- [x] **Step 2: Run focused S11 tests and verify GREEN**

Run:

```powershell
pytest tests/test_s11_frequency_plots.py -q
```

Expected: all S11 tests pass.

### Task 3: Expose plot workers only for one-folder runs

**Files:**
- Modify: `deflector_tuning/runner.py`
- Modify: `run_folder_analysis.py`
- Modify: `run_all_folder_analyses.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_run_folder_analysis_cli.py`
- Modify: `tests/test_run_all_folder_analyses_cli.py`

**Interfaces:**
- `run_folder_analysis(..., plot_workers: int = 1)` forwards `plot_workers` to `plot_s11_with_markers`.
- `run_folder_analysis.py` adds `--plot-workers`; values below one normalize to one.
- `run_all_folder_analyses.py` retains dataset-level `--workers` and calls `run_folder_analysis(..., plot_workers=1)` to prevent nested pools.

- [x] **Step 1: Add failing CLI/runner assertions**

Assert `--plot-workers 3` parses as `3`, nonpositive values normalize to `1`, and batch task calls use serial plot rendering.

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py tests/test_runner.py -q
```

Expected: failures for the missing argument/forwarding behavior.

- [x] **Step 3: Implement CLI and runner wiring**

Add the argument, normalize it in `apply_inferred_defaults`, pass it through `main`, add the runner parameter, and explicitly pass `plot_workers=1` from `run_batch_task`.

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
pytest tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py tests/test_runner.py -q
```

Expected: all focused runner and CLI tests pass.

### Task 4: Update usage documentation and verify the full repository

**Files:**
- Modify: `README.md` or the existing relevant usage document after locating the current one-folder command section.

- [x] **Step 1: Document the command**

Document:

```powershell
python run_folder_analysis.py sim/<dataset> --plot-workers 4
```

and explain that `run_all_folder_analyses.py --workers N` remains dataset-level multiprocessing.

- [x] **Step 2: Run the full verification chain**

Run:

```powershell
pytest -q
ruff check deflector_tuning/visualization/s11_frequency_plots.py deflector_tuning/runner.py run_folder_analysis.py run_all_folder_analyses.py tests/test_s11_frequency_plots.py tests/test_runner.py tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py
git diff --check
```

Expected: zero test failures, zero Ruff errors, and no whitespace errors.
