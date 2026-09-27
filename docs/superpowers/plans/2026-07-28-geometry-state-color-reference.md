# Geometry-response State Color and Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distinguish Cell and Iris traces with light/dark variants of each marker hue, mark the sampled geometry baseline, and simplify phase-axis labels.

**Architecture:** Keep all changes inside the existing geometry-response plotting module. Derive the Cell tint from the shared marker base color, retain the base color for Iris, and draw the reference from the response table's `sweep_base` column so no dataset-specific branch is introduced.

**Tech Stack:** Python, pandas, Matplotlib, pytest

## Global Constraints

- Preserve the red, blue, and green frequency-marker hue families.
- Cell is lighter and less saturated; Iris is darker and more saturated.
- Preserve circle/solid for Cell and square/dashed for Iris.
- The pickup remains each marker/state's phase relative to itself at `sweep_base`.
- Do not change marker frequencies, numerical phase data, units, or other plot families.
- Do not commit because the repository guide requires explicit user authorization.

---

### Task 1: State-specific color variants

**Files:**
- Modify: `deflector_tuning/visualization/geometry_phase_response_plots.py`
- Test: `tests/test_geometry_phase_response.py`

**Interfaces:**
- Consumes: `MARKER_COLORS: dict[str, str]`
- Produces: `_cell_color(base_color: str) -> str` and state-specific colors in `_plot_family_lines`

- [ ] **Step 1: Write the failing test**

Capture the real Matplotlib figure through the existing public
`plot_geometry_phase_response` entry point. For each frequency pair, assert
that Cell and Iris colors differ, Iris retains the literal base color, and the
Cell RGB channels are closer to white.

- [ ] **Step 2: Run the color test and verify RED**

Run:
`python -m pytest tests/test_geometry_phase_response.py::test_plot_geometry_phase_response_uses_light_cell_and_saturated_iris_colors -q`

Expected: FAIL because both state traces currently use the same color.

- [ ] **Step 3: Implement the minimal color transformation**

Blend each base RGB triplet 35% toward white for Cell, and pass the unchanged
base color to Iris. Keep the existing markers and line styles.

- [ ] **Step 4: Run the color test and verify GREEN**

Run the same targeted pytest command. Expected: PASS.

### Task 2: Baseline line and phase-axis labels

**Files:**
- Modify: `deflector_tuning/visualization/geometry_phase_response_plots.py`
- Test: `tests/test_geometry_phase_response.py`

**Interfaces:**
- Consumes: `response["sweep_base"]`
- Produces: one non-legend vertical line per figure and y-axis labels `$\phi$ [deg]` / `$\Delta\phi$ [deg]`

- [ ] **Step 1: Write the failing test**

Capture the absolute and pickup axes from the public plotting entry point.
Assert that each includes an `_sweep_reference` line whose x-data are both
`29.148`, that this label is absent from the legend, and that the two y-axis
labels are the approved math-text strings.

- [ ] **Step 2: Run the reference test and verify RED**

Run:
`python -m pytest tests/test_geometry_phase_response.py::test_plot_geometry_phase_response_marks_baseline_and_uses_phase_symbols -q`

Expected: FAIL because the baseline line and approved labels are absent.

- [ ] **Step 3: Implement the reference treatment**

Add `sweep_base` to the required table contract. Draw a gray dotted line at
the single sampled baseline with the private label `_sweep_reference`, and
apply the approved y-axis labels.

- [ ] **Step 4: Run the reference test and verify GREEN**

Run the same targeted pytest command. Expected: PASS.

### Task 3: Regenerate and verify the RF figures

**Files:**
- Regenerate: `fig/analyses/sim_sweep_260728_zlen/figures/geometry_phase_response/*.png`

**Interfaces:**
- Consumes: `fig/analyses/sim_sweep_260728_zlen/tables/geom_phase.csv`
- Produces: eight refreshed PNG files

- [ ] **Step 1: Run focused tests**

Run:
`python -m pytest tests/test_geometry_phase_response.py -q`

Expected: all tests pass.

- [ ] **Step 2: Regenerate through the existing plot function**

Read `geom_phase.csv` and call `plot_geometry_phase_response` with the
established output directory. Expected: eight paths returned.

- [ ] **Step 3: Inspect the combined figures**

Open `absolute_phase.png` and `phase_pickup.png`. Confirm light Cell traces,
saturated Iris traces, aligned legend columns, the 29.148 mm vertical line,
and unclipped labels.

- [ ] **Step 4: Run full verification**

Run:
`python -m pytest -q`

Then run `git diff --check` for the two modified source/test files and verify
all eight PNG files are nonempty.
