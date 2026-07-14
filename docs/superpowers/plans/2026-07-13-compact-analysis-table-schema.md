# Compact Analysis Table Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standard marker-analysis tables with one concise internal and persisted schema, safely refresh legacy CSVs in place, and add table-only batch regeneration.

**Architecture:** A central schema module owns canonical table keys, filenames, source metadata aliases, and forbidden legacy names. Analysis producers emit canonical columns directly; plots and scripts consume them directly. The save boundary projects constant metadata into manifest context, stages all managed CSVs before replacement, and removes only registered legacy filenames after a successful commit.

**Tech Stack:** Python 3.11+, pandas, matplotlib, pytest, pathlib, JSON manifests, PowerShell execution on Windows.

## Global Constraints

- Phase 1 covers only the 13 standard tables returned by `build_marker_analysis()`.
- Standard results remain under `fig/analyses/<dataset_id>/tables`; no parallel v2 directory is created.
- Internal DataFrames and CSVs use the same canonical column names.
- Canonical admittance columns contain `admit`; do not replace admittance with a one-letter `y` token.
- Existing concise shared columns such as `dataset_id`, `source_file`, `freq_ghz`, `s_db`, `s_phase_deg`, `sim_r_c`, and `sim_w_c` remain unchanged.
- Distinct `sim_*` source parameters are never merged because their current values happen to match.
- Dataset-specific schema branches are forbidden.
- Special Y11, dispersion, profile, and manually curated `outputs/*` schemas are not renamed in Phase 1.
- Legacy cleanup touches only filenames registered as former standard filenames.
- Implementation follows red-green-refactor; every production behavior begins with a failing test.

---

### Task 1: Canonical Table Contract

**Files:**
- Create: `deflector_tuning/table_schema.py`
- Create: `tests/test_table_schema.py`

**Interfaces:**
- Produces: `TableNameSpec`, `STANDARD_TABLE_SPECS`, `SOURCE_COLUMN_ALIASES`, `FORBIDDEN_LEGACY_COLUMNS`, `canonical_source_column(name: str) -> str`.
- Consumes: no production interfaces.

- [ ] **Step 1: Write the failing schema contract tests**

```python
from deflector_tuning.table_schema import (
    FORBIDDEN_LEGACY_COLUMNS,
    SOURCE_COLUMN_ALIASES,
    STANDARD_TABLE_SPECS,
    canonical_source_column,
)


def test_standard_table_specs_define_canonical_and_legacy_filenames() -> None:
    assert list(STANDARD_TABLE_SPECS) == [
        "markers",
        "marker_pts",
        "phase_polar",
        "kyhl_admit_audit",
        "kyhl_admit_pts",
        "kyhl_admit_steps",
        "cell_iris_cmp",
        "coupler_params",
        "rc_line",
        "phase_adv",
        "phase_stats",
        "nodal_shift",
        "geom_phase",
    ]
    assert STANDARD_TABLE_SPECS["marker_pts"].filename == "marker_pts.csv"
    assert STANDARD_TABLE_SPECS["marker_pts"].legacy_filename == "marker_points.csv"
    assert STANDARD_TABLE_SPECS["markers"].legacy_filename == "markers.csv"


def test_source_aliases_are_global_column_rules() -> None:
    assert SOURCE_COLUMN_ALIASES == {
        "sim_tuner_insertion_depth": "sim_tuner_depth",
        "sim_coupler_path_bot2_width": "sim_cpl_bot2_w",
    }
    assert canonical_source_column("sim_tuner_insertion_depth") == "sim_tuner_depth"
    assert canonical_source_column("sim_r_c") == "sim_r_c"


def test_forbidden_legacy_columns_include_duplicate_kyhl_aliases() -> None:
    assert "admittance_real" in FORBIDDEN_LEGACY_COLUMNS
    assert "kyhl_operation_real" in FORBIDDEN_LEGACY_COLUMNS
    assert "phase_advance_0to360_deg" in FORBIDDEN_LEGACY_COLUMNS
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_table_schema.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'deflector_tuning.table_schema'`.

- [ ] **Step 3: Implement the schema contract**

```python
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class TableNameSpec:
    filename: str
    legacy_filename: str


STANDARD_TABLE_SPECS = OrderedDict(
    [
        ("markers", TableNameSpec("markers.csv", "markers.csv")),
        ("marker_pts", TableNameSpec("marker_pts.csv", "marker_points.csv")),
        ("phase_polar", TableNameSpec("phase_polar.csv", "marker_phase_polar.csv")),
        ("kyhl_admit_audit", TableNameSpec("kyhl_admit_audit.csv", "kyhl_f2pi3_normalized_admittance_audit.csv")),
        ("kyhl_admit_pts", TableNameSpec("kyhl_admit_pts.csv", "kyhl_admittance_points.csv")),
        ("kyhl_admit_steps", TableNameSpec("kyhl_admit_steps.csv", "kyhl_admittance_transitions.csv")),
        ("cell_iris_cmp", TableNameSpec("cell_iris_cmp.csv", "cell_iris_response_comparison.csv")),
        ("coupler_params", TableNameSpec("coupler_params.csv", "coupler_cavity_parameter_estimates.csv")),
        ("rc_line", TableNameSpec("rc_line.csv", "grid_rc_line_scan.csv")),
        ("phase_adv", TableNameSpec("phase_adv.csv", "phase_advance.csv")),
        ("phase_stats", TableNameSpec("phase_stats.csv", "phase_summary.csv")),
        ("nodal_shift", TableNameSpec("nodal_shift.csv", "nodal_shift.csv")),
        ("geom_phase", TableNameSpec("geom_phase.csv", "geometry_phase_response.csv")),
    ]
)

SOURCE_COLUMN_ALIASES = {
    "sim_tuner_insertion_depth": "sim_tuner_depth",
    "sim_coupler_path_bot2_width": "sim_cpl_bot2_w",
}

FORBIDDEN_LEGACY_COLUMNS = frozenset(
    {
        "admittance_real",
        "admittance_imag",
        "kyhl_operation_real",
        "kyhl_operation_imag",
        "kyhl_operation_angle_deg",
        "phase_advance_0to360_deg",
        "phase_error_from_target_deg",
        "abs_phase_error_from_target_deg",
    }
)


def canonical_source_column(name: str) -> str:
    return SOURCE_COLUMN_ALIASES.get(name, name)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_table_schema.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the contract**

```powershell
git add deflector_tuning/table_schema.py tests/test_table_schema.py
git commit -m "refactor: define compact table schema contract"
```

---

### Task 2: Canonical Phase and Geometry Columns

**Files:**
- Modify: `deflector_tuning/analysis/phase_advance.py`
- Modify: `deflector_tuning/analysis/phase_summary.py`
- Modify: `deflector_tuning/analysis/nodal_shift.py`
- Modify: `deflector_tuning/analysis/geometry_phase_response.py`
- Modify: `deflector_tuning/visualization/phase_advance_plots.py`
- Modify: `deflector_tuning/visualization/nodal_shift_plots.py`
- Modify: `deflector_tuning/visualization/geometry_phase_response_plots.py`
- Modify: `scripts/build_revised_kyhl_validation_metrics.py`
- Modify: `tests/test_phase_advance.py`
- Modify: `tests/test_phase_summary.py`
- Modify: `tests/test_nodal_shift.py`
- Modify: `tests/test_phase_advance_plots.py`
- Modify: `tests/test_nodal_shift_plots.py`
- Modify: `tests/test_geometry_phase_response.py`

**Interfaces:**
- Consumes: `canonical_source_column()` for propagated source metadata.
- Produces: canonical phase columns consumed by the marker pipeline and plots.

- [ ] **Step 1: Change phase-family assertions to canonical names**

Use this exact mapping in tests:

```python
PHASE_COLUMN_RENAMES = {
    "from_source_file": "file_from",
    "to_source_file": "file_to",
    "from_tune_position": "pos_from",
    "to_tune_position": "pos_to",
    "target_freq_ghz": "freq_target_ghz",
    "from_freq_ghz": "freq_ghz",
    "from_s_db": "s_db_from",
    "to_s_db": "s_db_to",
    "from_phase_deg": "phase_from_deg",
    "to_phase_deg": "phase_to_deg",
    "signed_phase_step_deg": "phase_step_deg",
    "phase_advance_0to360_deg": "phase_adv_deg",
    "phase_error_from_240_deg": "phase_err_240_deg",
    "transition_count": "n_steps",
    "mean_phase_advance_deg": "phase_adv_mean_deg",
    "mean_phase_error_deg": "phase_err_mean_deg",
    "mean_abs_phase_error_deg": "phase_err_abs_mean_deg",
    "rms_phase_error_deg": "phase_err_rms_deg",
    "max_abs_phase_error_deg": "phase_err_abs_max_deg",
    "worst_from_tune_position": "worst_pos_from",
    "worst_to_tune_position": "worst_pos_to",
    "worst_phase_error_deg": "worst_phase_err_deg",
    "target_phase_advance_deg": "phase_target_deg",
    "phase_error_from_target_deg": "phase_err_deg",
    "abs_phase_error_from_target_deg": "phase_err_abs_deg",
}
```

Same-marker transitions keep one `freq_ghz`; tests assert `from_freq_ghz` and `to_freq_ghz` are absent.

- [ ] **Step 2: Run phase and geometry tests and verify RED**

Run: `python -m pytest tests/test_phase_advance.py tests/test_phase_summary.py tests/test_nodal_shift.py tests/test_geometry_phase_response.py -q`

Expected: assertion failures showing old columns.

- [ ] **Step 3: Update the four producers**

Apply the mapping above directly to output constants, row dictionaries, required-column checks, grouping, sorting, and error messages. Collapse the two equal transition-frequency columns to `freq_ghz`. Canonicalize propagated simulation metadata with `canonical_source_column()` and reject alias collisions.

Geometry response uses:

```python
GEOMETRY_COLUMN_RENAMES = {
    "baseline_sweep_value": "sweep_base",
    "cell_source_file": "cell_file",
    "iris_source_file": "iris_file",
    "cell_tune_position": "cell_pos",
    "iris_tune_position": "iris_pos",
    "cell_phase_shift_from_baseline_deg": "cell_phase_shift_deg",
    "iris_phase_shift_from_baseline_deg": "iris_phase_shift_deg",
    "phase_delta_cell_to_iris_deg": "cell_iris_phase_delta_deg",
    "phase_delta_shift_from_baseline_deg": "cell_iris_delta_shift_deg",
}
```

- [ ] **Step 4: Update plot and script consumers**

Replace every legacy lookup with the canonical name. `build_revised_kyhl_validation_metrics.py` reads `phase_adv_deg`, `phase_step_deg`, `phase_err_rms_deg`, `pos_from`, and `pos_to` from refreshed tables.

- [ ] **Step 5: Run focused producer and plot tests**

Run: `python -m pytest tests/test_phase_advance.py tests/test_phase_summary.py tests/test_nodal_shift.py tests/test_geometry_phase_response.py tests/test_phase_advance_plots.py tests/test_nodal_shift_plots.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the phase schema**

```powershell
git add deflector_tuning/analysis deflector_tuning/visualization scripts/build_revised_kyhl_validation_metrics.py tests
git commit -m "refactor: compact phase analysis columns"
```

---

### Task 3: Canonical Admittance, Comparison, and Coupler Columns

**Files:**
- Modify: `deflector_tuning/analysis/kyhl_admittance.py`
- Modify: `deflector_tuning/analysis/cell_iris_response.py`
- Modify: `deflector_tuning/analysis/coupler_cavity_parameters.py`
- Modify: `deflector_tuning/visualization/kyhl_admittance_plots.py`
- Modify: `deflector_tuning/visualization/cell_iris_response_plots.py`
- Modify: `deflector_tuning/visualization/coupler_cavity_parameter_plots.py`
- Modify: `scripts/build_kyhl_admittance_operation_metrics.py`
- Modify: `tests/test_kyhl_admittance.py`
- Modify: `tests/test_cell_iris_response.py`
- Modify: `tests/test_cell_iris_response_plots.py`
- Modify: `tests/test_coupler_cavity_parameters.py`
- Modify: `tests/test_coupler_cavity_parameter_plots.py`

**Interfaces:**
- Consumes: canonical phase columns from Task 2.
- Produces: compact admittance and comparison tables consumed by the marker pipeline and plots.

- [ ] **Step 1: Write failing canonical admittance assertions**

Tests require these point columns:

```python
POINT_COLUMNS = {
    "gamma_real": "gamma_re",
    "gamma_imag": "gamma_im",
    "operation_mode_deg": "op_mode_deg",
    "operation_mode_scale": "op_admit_scale",
    "operation_axes_deg": "op_admit_axes_deg",
    "operation_scaled_admittance_real": "op_admit_re",
    "operation_scaled_admittance_imag": "op_admit_im",
    "operation_scaled_admittance_abs": "op_admit_mag",
    "operation_scaled_admittance_angle_deg": "op_admit_ang_deg",
    "nearest_operation_axis_deg": "op_admit_axis_deg",
    "operation_axis_error_deg": "op_admit_axis_err_deg",
    "abs_operation_axis_error_deg": "op_admit_axis_err_abs_deg",
}
```

Tests assert the point and transition outputs omit `raw_pseudo_admittance_*`, generic `admittance_*`, and `kyhl_operation_*` aliases.

The audit uses `ref_admit_siemens`, `norm_imp_re`, `norm_imp_im`, `line_admit_re`, `line_admit_im`, `line_admit_mag`, `phys_admit_re_siemens`, `phys_admit_im_siemens`, `mode_admit_scale`, `mode_admit_re`, `mode_admit_im`, `mode_admit_mag`, `mode_admit_ang_deg`, and `mode_gamma_*`. It omits the duplicate line-admittance angle.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_kyhl_admittance.py tests/test_cell_iris_response.py tests/test_coupler_cavity_parameters.py -q`

Expected: assertion failures on legacy output names.

- [ ] **Step 3: Update KYHL producers and remove aliases**

Emit only the canonical operation coordinate in point and transition outputs. Keep raw reflection as `gamma_*`. Preserve the separate line, mode, and physical coordinates in the audit. Use `freq_ghz` once for same-marker transitions.

- [ ] **Step 4: Compact comparison and coupler outputs**

Use these naming patterns consistently:

```python
COMPARISON_COLUMN_RENAMES = {
    "cell_operation_scaled_admittance_delta_abs": "cell_admit_delta_mag",
    "iris_operation_scaled_admittance_delta_abs": "iris_admit_delta_mag",
    "operation_scaled_admittance_response_ratio_iris_over_cell": "admit_ratio_iris_cell",
    "phase_step_response_ratio_iris_over_cell": "phase_ratio_iris_cell",
    "cell_phase_residual_from_target_deg": "cell_phase_err_deg",
    "iris_phase_residual_from_target_deg": "iris_phase_err_deg",
    "target_external_quality_factor": "q_ext_target",
    "external_quality_factor": "q_ext",
    "reference_target_frequency_ghz": "ref_freq_target_ghz",
    "operation_target_frequency_ghz": "op_freq_target_ghz",
}
```

Boolean support fields use `supports_iris_admit`, `supports_iris_axis`, and `supports_iris_phase` without changing their meaning.

- [ ] **Step 5: Update plots, scripts, and tests**

Replace legacy column lookups and expected output lists. No consumer may create a compatibility alias.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_kyhl_admittance.py tests/test_cell_iris_response.py tests/test_cell_iris_response_plots.py tests/test_coupler_cavity_parameters.py tests/test_coupler_cavity_parameter_plots.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the admittance schema**

```powershell
git add deflector_tuning/analysis deflector_tuning/visualization scripts/build_kyhl_admittance_operation_metrics.py tests
git commit -m "refactor: compact admittance analysis columns"
```

---

### Task 4: Canonical Analysis Table Keys

**Files:**
- Modify: `deflector_tuning/analysis/marker_pipeline.py`
- Modify: `deflector_tuning/runner.py`
- Modify: `deflector_tuning/workflows/mode_detection.py`
- Modify: `deflector_tuning/visualization/polar_phase_views.py`
- Modify: `deflector_tuning/visualization/grid_scan_phase_line_plots.py`
- Modify: `deflector_tuning/visualization/grid_scan_spacing_maps.py`
- Modify: `deflector_tuning/visualization/s11_frequency_plots.py`
- Modify: `tests/test_marker_analysis_pipeline.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_polar_phase_views.py`
- Modify: `tests/test_grid_scan_phase_line_plots.py`
- Modify: `tests/test_grid_scan_spacing_maps.py`
- Modify: `tests/test_s11_frequency_plots.py`

**Interfaces:**
- Consumes: `STANDARD_TABLE_SPECS` and canonical outputs from Tasks 2-3.
- Produces: an `AnalysisTables` mapping with exactly the 13 canonical keys.

- [ ] **Step 1: Change pipeline tests to canonical keys**

Assert the exact ordered key list from `STANDARD_TABLE_SPECS` and update all result lookups such as `result["marker_pts"]`, `result["phase_adv"]`, and `result["kyhl_admit_pts"]`.

- [ ] **Step 2: Run pipeline and runner tests and verify RED**

Run: `python -m pytest tests/test_marker_analysis_pipeline.py tests/test_runner.py -q`

Expected: failures showing legacy table keys.

- [ ] **Step 3: Rename producer keys and all consumers**

Return the canonical ordered keys from `build_marker_analysis()`. Update runner plot inputs, mode detection, and every visualization lookup. Keep figure group names unchanged where they describe the figure rather than a table artifact.

- [ ] **Step 4: Apply source metadata aliases at load propagation**

Canonicalize known source metadata columns before downstream analysis. Raise `ValueError` when an input already contains both a legacy source name and its canonical alias.

- [ ] **Step 5: Run all affected tests**

Run: `python -m pytest tests/test_marker_analysis_pipeline.py tests/test_runner.py tests/test_polar_phase_views.py tests/test_grid_scan_phase_line_plots.py tests/test_grid_scan_spacing_maps.py tests/test_s11_frequency_plots.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the table-key migration**

```powershell
git add deflector_tuning tests
git commit -m "refactor: rename standard analysis tables"
```

---

### Task 5: Compact CSV Projection, Manifest Context, and Safe Cleanup

**Files:**
- Create: `deflector_tuning/table_export.py`
- Modify: `deflector_tuning/analysis/marker_pipeline.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `deflector_tuning/runner.py`
- Create: `tests/test_table_export.py`
- Modify: `tests/test_marker_analysis_pipeline.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Produces: `TableSaveResult(paths: AnalysisPaths, constants: dict[str, dict[str, object]])` and `save_standard_tables(tables, output_dir) -> TableSaveResult`.
- Consumes: `STANDARD_TABLE_SPECS` and canonical `AnalysisTables`.

- [ ] **Step 1: Write failing projection and transaction tests**

Cover these behaviors with real temporary files:

```python
def test_projection_keeps_constant_results_and_moves_constant_metadata_to_context() -> None:
    table = pd.DataFrame(
        {
            "dataset_id": ["sim_sweep_260713_case"] * 2,
            "marker_role": ["sim"] * 2,
            "op_admit_mag": [1.5, 1.5],
        }
    )
    projected, constants = project_table("kyhl_admit_pts", table)
    assert list(projected) == ["dataset_id", "op_admit_mag"]
    assert constants == {"marker_role": "sim"}


def test_projection_keeps_varying_sim_sweep_columns() -> None:
    table = pd.DataFrame(
        {
            "dataset_id": ["sim_grid_260626_case"] * 2,
            "source_file": ["a.s1p", "b.s1p"],
            "sim_r_c": [55.5, 56.5],
            "s_db": [-20.0, -21.0],
        }
    )
    projected, constants = project_table("marker_pts", table)
    assert "sim_r_c" in projected
    assert "sim_r_c" not in constants


def test_empty_table_omits_context_headers() -> None:
    table = pd.DataFrame(columns=["dataset_id", "marker_name", "marker_role", "phase_err_rms_deg"])
    projected, constants = project_table("phase_stats", table)
    assert list(projected) == ["dataset_id", "marker_name", "phase_err_rms_deg"]
    assert constants == {}


def test_successful_save_removes_registered_legacy_files_only(tmp_path: Path) -> None:
    (tmp_path / "marker_points.csv").write_text("old\n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("user\n", encoding="utf-8")
    result = save_standard_tables(_minimal_standard_tables(), tmp_path)
    assert result.paths["marker_pts"] == tmp_path / "marker_pts.csv"
    assert not (tmp_path / "marker_points.csv").exists()
    assert (tmp_path / "notes.csv").read_text(encoding="utf-8") == "user\n"


def test_failed_staging_preserves_existing_managed_files(monkeypatch, tmp_path: Path) -> None:
    legacy = tmp_path / "marker_points.csv"
    legacy.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        save_standard_tables(_minimal_standard_tables(), tmp_path)
    assert legacy.read_text(encoding="utf-8") == "old\n"
```

Use `dataset_id` and table-specific row identifiers as required identities. Context candidates include `data_kind`, `data_layer`, `marker_role`, `port_side`, `s_name`, `scan_type`, marker-source metadata, temperature/environment metadata, calculation convention fields, and every `sim_*` column. Result columns are explicitly registered per table and never projected out.

- [ ] **Step 2: Run export tests and verify RED**

Run: `python -m pytest tests/test_table_export.py -q`

Expected: collection fails because `deflector_tuning.table_export` does not exist.

- [ ] **Step 3: Implement projection and scalar normalization**

```python
CONTEXT_COLUMNS = frozenset(
    {
        "data_kind",
        "data_layer",
        "marker_role",
        "port_side",
        "s_name",
        "scan_type",
        "marker_source",
        "source_format",
        "frequency_scale_factor",
        "temp_op_C",
        "temp_meas_C",
        "humidity_fraction",
        "reference_ohm",
        "is_normalized",
        "op_mode_deg",
        "op_admit_scale",
        "op_admit_axes_deg",
        "axis_sign",
        "convention_note",
        "formula_note",
        "source_note",
    }
)


@dataclass(frozen=True)
class TableSaveResult:
    paths: OrderedDict[str, Path]
    constants: dict[str, dict[str, object]]


def project_table(table_key: str, table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if table_key not in STANDARD_TABLE_SPECS:
        raise KeyError(f"Unknown standard table: {table_key}")
    keep: list[str] = []
    constants: dict[str, object] = {}
    for column in table.columns:
        is_context = column in CONTEXT_COLUMNS or column.startswith("sim_")
        if column == "dataset_id" or not is_context:
            keep.append(column)
            continue
        values = table[column].dropna()
        if values.empty:
            continue
        if values.nunique(dropna=True) == 1:
            constants[column] = json_scalar(values.iloc[0])
            continue
        keep.append(column)
    return table.loc[:, keep].copy(), constants


def json_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported manifest scalar: {type(value).__name__}")
```

- [ ] **Step 4: Implement staged replacement and rollback**

Serialize all 13 canonical tables to a staging directory inside `tables`. Validate key order, filenames, and headers. Back up current managed canonical and legacy files, replace canonical files, restore backups on any replacement error, and delete differing legacy filenames only after all canonical files exist. Preserve unrelated files.

- [ ] **Step 5: Extend manifest writing**

Add `table_schema_version=2` and `table_constants` to standard marker-analysis manifests. Write manifests through a temporary sibling followed by `Path.replace()`.

- [ ] **Step 6: Run export, pipeline, and runner tests**

Run: `python -m pytest tests/test_table_export.py tests/test_marker_analysis_pipeline.py tests/test_runner.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit safe persistence**

```powershell
git add deflector_tuning tests
git commit -m "feat: persist compact analysis tables safely"
```

---

### Task 6: Table-Only Single and Batch Runs

**Files:**
- Modify: `run_folder_analysis.py`
- Modify: `run_all_folder_analyses.py`
- Modify: `deflector_tuning/runner.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `tests/test_run_folder_analysis_cli.py`
- Modify: `tests/test_run_all_folder_analyses_cli.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Produces: CLI flag `--tables-only`; function parameter `tables_only: bool = False`; batch task field `tables_only: bool`.
- Consumes: safe standard table save result from Task 5.

- [ ] **Step 1: Write failing CLI and runner tests**

Tests assert:

```python
assert "--tables-only" in help_result.stdout
assert captured_run_kwargs["tables_only"] is True
assert existing_figure_path.exists()
assert manifest["outputs"]["figures"] == existing_figure_manifest
assert manifest["table_schema_version"] == 2
```

Batch tests confirm every `BatchTask` receives the flag and successful standard tasks contain canonical filenames without differing legacy filenames.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py tests/test_runner.py -q`

Expected: parser and keyword assertion failures because `--tables-only` is absent.

- [ ] **Step 3: Add the one-dataset flag and runner behavior**

When `tables_only=True`, build and save analysis tables, detect modes, skip plot functions, preserve only existing figure paths from the previous manifest, and atomically refresh the manifest.

- [ ] **Step 4: Add batch propagation and verification**

Add the flag to `BatchTask`, parser, task construction, and `run_batch_task()`. Verification uses `STANDARD_TABLE_SPECS`; it never deletes files itself because cleanup belongs to the transactional saver.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py tests/test_runner.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit table-only execution**

```powershell
git add run_folder_analysis.py run_all_folder_analyses.py deflector_tuning tests
git commit -m "feat: add table-only batch refresh"
```

---

### Task 7: Repository Verification and Existing Result Refresh

**Files:**
- Verify: `deflector_tuning/**`, `scripts/**`, `run_folder_analysis.py`, `run_all_folder_analyses.py`, and `tests/**`.
- Regenerate: `fig/analyses/<dataset_id>/tables/*.csv` in the main project data workspace; these generated outputs remain outside the feature commit unless already tracked.

**Interfaces:**
- Consumes: every completed task.
- Produces: verified code and refreshed existing batch-managed standard tables.

- [ ] **Step 1: Scan executable code for forbidden names**

Run:

```powershell
rg -n "marker_points|marker_phase_polar|kyhl_f2pi3_normalized_admittance_audit|kyhl_admittance_points|kyhl_admittance_transitions|cell_iris_response_comparison|coupler_cavity_parameter_estimates|grid_rc_line_scan|phase_advance_0to360_deg|phase_error_from_target_deg|geometry_phase_response" deflector_tuning scripts run_folder_analysis.py run_all_folder_analyses.py
```

Expected: no executable legacy schema lookup; function names and historical migration registries may remain only where explicitly allowed.

- [ ] **Step 2: Run the complete test suite**

Run: `python -m pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 3: Run representative table-only smoke analyses**

Run one valid sweep dataset and one valid grid dataset against temporary output roots. Compare canonical row counts and mapped numeric values against their existing legacy tables.

Expected: row counts match; mapped numeric values match within existing floating-point tolerances; canonical filenames and manifest version 2 are present.

- [ ] **Step 4: Refresh existing valid batch-managed tables**

Run `run_all_folder_analyses.py --tables-only` with the main project `data` and `fig/analyses` roots, using explicit valid dataset IDs when strict dataset naming rejects unrelated legacy folders.

Expected: each successful standard marker dataset prints its input/output mapping; failures remain listed and return a nonzero status without cleaning failed dataset directories.

- [ ] **Step 5: Audit refreshed results**

Check every successfully refreshed standard marker directory for:

```text
13 canonical managed filenames
0 differing legacy managed filenames
0 forbidden legacy headers
manifest table_schema_version = 2
manifest table_constants present
unrelated CSVs preserved
```

- [ ] **Step 6: Run final repository verification**

Run:

```powershell
git diff --check
python -m pytest -q
git status --short
```

Expected: no whitespace errors, all tests pass, and only intended source/test/doc changes remain.

- [ ] **Step 7: Commit final integration fixes**

```powershell
git add deflector_tuning run_folder_analysis.py run_all_folder_analyses.py scripts tests docs
git commit -m "refactor: complete compact analysis table migration"
```
