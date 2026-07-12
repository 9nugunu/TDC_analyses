# Navigator-driven S11 Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one readable S11 figure per distinct changing simulation Navigator parameter point and omit simulation sweep overviews.

**Architecture:** Extend the existing shared simulation grouping helper so `sim_r_c` and `sim_w_c` participate in the same variable-column detection as all other Navigator metadata. The current grid-point planner remains responsible for two-dimensional grid compatibility, while one-dimensional Navigator sweeps use the existing generic simulation-sweep planner; both generate human-readable titles from shared parameter formatting.

**Tech Stack:** Python 3, pandas, matplotlib, pytest.

## Global Constraints

- Do not rename input files or change `result_navigator.csv` parsing.
- Preserve raw-data and simulation-without-changing-Navigator overview behavior.
- Use Navigator metadata values rather than parsing source filenames.
- Do not commit unless the user explicitly asks.

---

### Task 1: Define parameter-point grouping behavior with tests

**Files:**
- Modify: `tests/test_s11_frequency_plots.py`
- Modify: `deflector_tuning/visualization/s11_frequency_plots.py`

**Interfaces:**
- Consumes: S11 and marker DataFrames with `source_file`, required S11 columns, and optional `sim_*` metadata.
- Produces: `build_s11_plot_plans(...) -> list[S11PlotPlan]` whose simulation plans contain no `overview` entry when Navigator metadata varies.

- [ ] **Step 1: Write the failing one-dimensional Navigator test**

```python
def test_build_s11_plot_plans_splits_one_dimensional_navigator_points(tmp_path: Path) -> None:
    s_table = _sparameter_table().assign(source_file=["run_001.s1p"] * 3 + ["run_002.s1p"] * 3, tune_position=pd.NA, sim_r_c=[56.09] * 3 + [56.10] * 3)
    marker_points = _marker_points().assign(source_file=["run_001.s1p", "run_002.s1p"], tune_position=pd.NA, sim_r_c=[56.09, 56.10])

    plans = build_s11_plot_plans(s_table, marker_points, tmp_path)

    assert [plan.key for plan in plans] == ["r_c_56p09", "r_c_56p10"]
    assert [plan.output_path.name for plan in plans] == ["r_c_56p09.png", "r_c_56p10.png"]
    assert [plan.title for plan in plans] == [
        "S11 magnitude | r_c = 56.09 mm",
        "S11 magnitude | r_c = 56.1 mm",
    ]
```

- [ ] **Step 2: Run the test to verify the current behavior fails**

Run: `pytest tests/test_s11_frequency_plots.py::test_build_s11_plot_plans_splits_one_dimensional_navigator_points -q`

Expected: FAIL because the current planner returns an `overview` plan for a one-dimensional `sim_r_c` sweep.

- [ ] **Step 3: Implement shared changing-column detection**

In `deflector_tuning/visualization/simulation_grouping.py`, change
`varying_sim_sweep_columns` so it includes every `sim_*` column whose
non-null values vary, except `sim_Num*` bookkeeping fields. This makes
`sim_r_c` and `sim_w_c` ordinary Navigator dimensions while retaining the
existing NumDepth handling in S11 planning.

- [ ] **Step 4: Route one-dimensional Navigator sweeps through the existing parameter-point planner**

In `deflector_tuning/visualization/s11_frequency_plots.py`, keep the
grid-point branch for its existing depth and position compatibility. Let the
generic simulation-sweep branch use the now-inclusive changing columns:

```python
group_columns = _sim_sweep_group_columns(s_table)
for values, group in table.groupby(group_columns, dropna=False, sort=False):
    key = _format_simulation_point_key(group_columns, values, group)
    title = _format_simulation_point_title(group_columns, values)
    plans.append(S11PlotPlan(key, "sim_sweep", folder / f"{key}.png", title, group, marker_group))
```

The generic simulation-sweep title must use the same readable parameter
formatter as the grid path. The non-simulation fallback must remain the
current overview plus optional tune-position plans.

- [ ] **Step 5: Run the focused test to verify it passes**

Run: `pytest tests/test_s11_frequency_plots.py::test_build_s11_plot_plans_splits_one_dimensional_navigator_points -q`

Expected: PASS.

### Task 2: Preserve two-dimensional names and readable titles

**Files:**
- Modify: `tests/test_s11_frequency_plots.py`
- Modify: `deflector_tuning/visualization/s11_frequency_plots.py`

**Interfaces:**
- Consumes: two varying Navigator columns such as `sim_r_c` and `sim_w_c`.
- Produces: point-specific filename keys and titles with the parameter names and values.

- [ ] **Step 1: Write the failing two-dimensional title test**

```python
def test_build_s11_plot_plans_uses_navigator_values_in_two_dimensional_titles(tmp_path: Path) -> None:
    s_table = _sparameter_table().assign(
        source_file=["run_001.s1p"] * 3 + ["run_002.s1p"] * 3,
        tune_position=pd.NA,
        sim_r_c=[56.09] * 3 + [56.10] * 3,
        sim_w_c=[19.0224] * 3 + [19.1224] * 3,
    )
    marker_points = _marker_points().assign(
        source_file=["run_001.s1p", "run_002.s1p"],
        tune_position=pd.NA,
        sim_r_c=[56.09, 56.10],
        sim_w_c=[19.0224, 19.1224],
    )

    plan = build_s11_plot_plans(s_table, marker_points, tmp_path)[0]

    assert plan.output_path.name == "r_c_56p09_w_c_19p0224.png"
    assert plan.title == "S11 magnitude | r_c = 56.09 mm; w_c = 19.0224 mm"
```

- [ ] **Step 2: Run the test to verify the title expectation fails**

Run: `pytest tests/test_s11_frequency_plots.py::test_build_s11_plot_plans_uses_navigator_values_in_two_dimensional_titles -q`

Expected: FAIL because the current grid title is `r_c=..., w_c=...: S11 magnitude`.

- [ ] **Step 3: Add parameter display helpers**

Implement helpers that map `sim_r_c` to `r_c = <value> mm` and `sim_w_c` to
`w_c = <value> mm`; map every other `sim_*` column to a snake-case name plus
its formatted value without guessing a unit. Reuse the same ordered parts for
the filename key and title to prevent them from drifting.

Retain the current one-point `sim_r_c`/`sim_w_c` behavior: when both geometry
columns are populated but neither varies, generate one point-specific figure
rather than falling back to `with_markers.png`.

- [ ] **Step 4: Run focused S11 tests**

Run: `pytest tests/test_s11_frequency_plots.py -q`

Expected: PASS, including existing raw overview, port-side, grid, and
simulation-sweep coverage.

### Task 3: Verify output behavior and repository quality

**Files:**
- Modify only files changed by Tasks 1 and 2.

**Interfaces:**
- Produces: tested S11 plans and unchanged public `plot_s11_with_markers` API.

- [ ] **Step 1: Run simulation metadata tests**

Run: `pytest tests/test_sim_metadata.py tests/test_s11_frequency_plots.py -q`

Expected: PASS.

- [ ] **Step 2: Run static and whitespace checks**

Run: `ruff check deflector_tuning/visualization/s11_frequency_plots.py deflector_tuning/visualization/simulation_grouping.py tests/test_s11_frequency_plots.py && git diff --check`

Expected: exit code 0.

- [ ] **Step 3: Inspect the staged behavior without changing input data**

Run: `python run_folder_analysis.py sim_sweep_260701_1DRcFine --output-dir <temporary-output-dir>`

Expected: the temporary `figures/s11` directory contains one `r_c_<value>.png` file per Navigator point and no `with_markers.png`.

- [ ] **Step 4: Leave the changes uncommitted**

Do not run `git commit`. The repository policy requires an explicit user request before committing.
