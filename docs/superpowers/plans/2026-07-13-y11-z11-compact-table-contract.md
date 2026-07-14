# Y11/Z11 Compact Table Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compact, transactional Y11 and Z11 table lanes with `y11_pts.csv`/`z11_pts.csv`, shared manifest contracts, tables-only execution, and real result regeneration.

**Architecture:** A focused one-port matrix module loads and samples `.y1p` and `.z1p` data while exposing parameter-specific compact columns. The existing transactional table saver is generalized around named ordered contracts (`standard`, `y11`, `z11`), and runner/batch verification selects behavior from the contract declared in the manifest.

**Tech Stack:** Python 3.11+, pandas, matplotlib, pytest, pathlib, JSON manifests, Touchstone RI/DB input, PowerShell on Windows.

## Global Constraints

- Scope is limited to direct one-port `.y1p` and `.z1p` files representing Y11 and Z11.
- Canonical Y11 output is `y11_pts.csv` with `y_re_siemens` and `y_im_siemens`.
- Canonical Z11 output is `z11_pts.csv` with `z_re_ohm` and `z_im_ohm`.
- Shared marker-frequency column is `freq_target_ghz`, not `target_freq_ghz`.
- Y11 and Z11 remain separate contracts; Siemens and ohms are never mixed in a generic value column.
- The existing standard 13-table contract remains behaviorally unchanged.
- All lanes use staged replacement, rollback, constant-context projection, and registered legacy cleanup.
- A mixed `.y1p`/`.z1p` folder is rejected as ambiguous.
- Multi-port Y/Z files remain unsupported.
- `--tables-only` skips figure generation and preserves valid existing figure paths.
- Existing profile, dispersion, and manually curated `outputs/*` schemas remain outside scope.

---

### Task 1: Direct One-Port Y11/Z11 Loading and Sampling

**Files:**
- Create: `deflector_tuning/data_loading/one_port_matrix.py`
- Modify: `deflector_tuning/data_loading/readers/touchstone_reader.py`
- Modify: `deflector_tuning/data_loading/admittance.py`
- Modify: `tests/test_touchstone_reader.py`
- Modify: `tests/test_y_admittance_workflow.py`
- Create: `tests/test_one_port_matrix.py`

**Interfaces:**
- Produces: `detect_one_port_matrix_lane(path) -> Literal["y11", "z11"] | None`, `load_y11_touchstone_folder(path)`, `load_z11_touchstone_folder(path)`, `extract_one_port_marker_frequencies(path)`, `sample_y11_markers(table, markers)`, and `sample_z11_markers(table, markers)`.
- Consumes: `read_touchstone(path)` and the existing result-navigator metadata convention.

- [ ] **Step 1: Write failing Touchstone Z-suffix and routing tests**

```python
def test_read_touchstone_accepts_z1p(tmp_path: Path) -> None:
    path = tmp_path / "case.z1p"
    path.write_text("# GHz Z RI R 50\n2.85 12.0 -3.0\n", encoding="utf-8")
    data = read_touchstone(path)
    assert data.header.parameter == "Z"
    assert data.s_values == [[complex(12.0, -3.0)]]


def test_detect_one_port_matrix_lane_rejects_mixed_y_and_z(tmp_path: Path) -> None:
    (tmp_path / "a.y1p").write_text("# GHz Y RI R 1\n2.85 0.1 0.2\n", encoding="utf-8")
    (tmp_path / "b.z1p").write_text("# GHz Z RI R 50\n2.85 10 20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both .y1p and .z1p"):
        detect_one_port_matrix_lane(tmp_path)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_touchstone_reader.py tests/test_one_port_matrix.py -q`

Expected: `.z1p` is rejected and `one_port_matrix` cannot be imported.

- [ ] **Step 3: Add explicit `.zNp` parsing and lane detection**

Update `_value_count_from_suffix()` to accept `(".s", ".y", ".z")`. Create:

```python
Lane = Literal["y11", "z11"]


def detect_one_port_matrix_lane(path: str | Path) -> Lane | None:
    folder = Path(path)
    has_y = any(p.is_file() and p.suffix.lower() == ".y1p" for p in folder.iterdir())
    has_z = any(p.is_file() and p.suffix.lower() == ".z1p" for p in folder.iterdir())
    if has_y and has_z:
        raise ValueError(f"Direct matrix folder contains both .y1p and .z1p files: {folder}")
    if has_y:
        return "y11"
    if has_z:
        return "z11"
    return None
```

- [ ] **Step 4: Write failing compact loader and sampler tests**

```python
def test_load_and_sample_z11_uses_compact_impedance_columns(tmp_path: Path) -> None:
    folder = _one_port_folder(tmp_path, suffix="z1p", parameter="Z", values=(12.0, -3.0))
    loaded = load_z11_touchstone_folder(folder)
    sampled = sample_z11_markers(loaded, _markers())
    assert {"z_re_ohm", "z_im_ohm", "freq_target_ghz"}.issubset(sampled.columns)
    assert "z_real_ohm" not in sampled
    assert sampled.iloc[0]["z_re_ohm"] == pytest.approx(12.0)


def test_load_and_sample_y11_renames_legacy_value_columns(tmp_path: Path) -> None:
    folder = _one_port_folder(tmp_path, suffix="y1p", parameter="Y", values=(0.01, 0.02))
    sampled = sample_y11_markers(load_y11_touchstone_folder(folder), _markers())
    assert {"y_re_siemens", "y_im_siemens", "freq_target_ghz"}.issubset(sampled.columns)
    assert "y_real_siemens" not in sampled
    assert "target_freq_ghz" not in sampled
```

- [ ] **Step 5: Run the loader tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_port_matrix.py tests/test_y_admittance_workflow.py -q`

Expected: missing Z functions and legacy Y column assertions fail.

- [ ] **Step 6: Implement shared one-port loading and sampling**

Use an internal immutable spec:

```python
@dataclass(frozen=True)
class OnePortMatrixSpec:
    lane: Lane
    parameter: str
    suffix: str
    element_column: str
    element_value: str
    real_column: str
    imag_column: str


SPECS = {
    "y11": OnePortMatrixSpec("y11", "Y", ".y1p", "y_name", "Y11", "y_re_siemens", "y_im_siemens"),
    "z11": OnePortMatrixSpec("z11", "Z", ".z1p", "z_name", "Z11", "z_re_ohm", "z_im_ohm"),
}
```

`load_one_port_matrix_folder()` validates the header parameter and one complex value per row. `extract_one_port_marker_frequencies()` owns the result-navigator marker extraction shared by both lanes. `sample_one_port_markers()` emits `freq_target_ghz` and `freq_error_ghz`, retaining navigator fields and varying `sim_*` metadata. Keep `extract_y11_marker_frequencies()` as a compatibility wrapper and re-export the existing Y11 loader/sampler names from `admittance.py`.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_touchstone_reader.py tests/test_one_port_matrix.py tests/test_y_admittance_workflow.py -q`

Expected: all selected tests pass.

- [ ] **Step 8: Commit if implementation commits were explicitly authorized**

```powershell
git add deflector_tuning/data_loading tests/test_touchstone_reader.py tests/test_one_port_matrix.py tests/test_y_admittance_workflow.py
git commit -m "feat: load compact Y11 and Z11 samples"
```

---

### Task 2: Named Transactional Table Contracts

**Files:**
- Modify: `deflector_tuning/table_schema.py`
- Modify: `deflector_tuning/table_export.py`
- Modify: `tests/test_table_schema.py`
- Modify: `tests/test_table_export.py`

**Interfaces:**
- Consumes: `TableNameSpec`, `project_table()`, and the existing standard table contract.
- Produces: `TABLE_CONTRACTS`, `table_specs(contract_name)`, `save_table_contract(tables, output_dir, contract_name)`, `save_y11_tables()`, and `save_z11_tables()`.

- [ ] **Step 1: Write failing contract registry tests**

```python
def test_direct_matrix_contracts_use_selected_names() -> None:
    assert list(TABLE_CONTRACTS["y11"]) == ["markers", "y11_pts"]
    assert TABLE_CONTRACTS["y11"]["y11_pts"].filename == "y11_pts.csv"
    assert TABLE_CONTRACTS["y11"]["y11_pts"].legacy_filename == "y11_marker_points.csv"
    assert list(TABLE_CONTRACTS["z11"]) == ["markers", "z11_pts"]
    assert TABLE_CONTRACTS["z11"]["z11_pts"].filename == "z11_pts.csv"
    assert TABLE_CONTRACTS["z11"]["z11_pts"].legacy_filename is None
```

- [ ] **Step 2: Run schema tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_table_schema.py -q`

Expected: `TABLE_CONTRACTS` is missing and `legacy_filename` does not allow `None`.

- [ ] **Step 3: Add the named contract registry**

Change `TableNameSpec.legacy_filename` to `str | None`. Preserve `STANDARD_TABLE_SPECS` as the same ordered object and add:

```python
Y11_TABLE_SPECS = OrderedDict([
    ("markers", TableNameSpec("markers.csv", "markers.csv")),
    ("y11_pts", TableNameSpec("y11_pts.csv", "y11_marker_points.csv")),
])
Z11_TABLE_SPECS = OrderedDict([
    ("markers", TableNameSpec("markers.csv", "markers.csv")),
    ("z11_pts", TableNameSpec("z11_pts.csv", None)),
])
TABLE_CONTRACTS = {
    "standard": STANDARD_TABLE_SPECS,
    "y11": Y11_TABLE_SPECS,
    "z11": Z11_TABLE_SPECS,
}
```

- [ ] **Step 4: Write failing Y11/Z11 persistence tests**

```python
def test_save_y11_tables_projects_context_and_removes_legacy(tmp_path: Path) -> None:
    (tmp_path / "y11_marker_points.csv").write_text("old\n", encoding="utf-8")
    result = save_table_contract(_y11_tables(), tmp_path, "y11")
    assert list(result.paths) == ["markers", "y11_pts"]
    assert (tmp_path / "y11_pts.csv").exists()
    assert not (tmp_path / "y11_marker_points.csv").exists()
    assert result.constants["y11_pts"]["y_name"] == "Y11"


def test_save_z11_tables_rolls_back_complete_contract_on_replace_error(monkeypatch, tmp_path: Path) -> None:
    old = tmp_path / "markers.csv"
    old.write_text("old\n", encoding="utf-8")
    _fail_second_replace(monkeypatch)
    with pytest.raises(OSError, match="replace failed"):
        save_table_contract(_z11_tables(), tmp_path, "z11")
    assert old.read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "z11_pts.csv").exists()
```

- [ ] **Step 5: Run export tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_table_export.py -q`

Expected: `save_table_contract` is missing.

- [ ] **Step 6: Generalize the saver around one explicit contract**

Implement:

```python
def save_table_contract(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    contract_name: str,
) -> TableSaveResult:
    specs = table_specs(contract_name)
    projected, constants = _project_contract_tables(tables, specs)
    staged = _stage_contract_csvs(projected, output_dir, specs)
    _validate_staged_contract(staged, projected, specs)
    backups = _backup_managed_contract_files(output_dir, specs)
    try:
        paths = _replace_contract_files(staged, output_dir, specs)
        _remove_registered_legacy_files(output_dir, specs)
    except Exception:
        _restore_managed_contract_files(output_dir, specs, backups)
        raise
    finally:
        _remove_contract_transaction_residue(output_dir, specs)
    return TableSaveResult(paths=paths, constants=constants)
```

The helper names may follow the existing module, but each helper receives `specs` and iterates only the active contract's canonical and non-`None` legacy filenames. Projection validates exact contract keys before staging. Add `y_name`, `z_name`, `source_format`, and `reference_ohm` to `CONTEXT_COLUMNS`; add `y_real_siemens`, `y_imag_siemens`, and `target_freq_ghz` to the legacy-header rejection set. Keep `save_standard_tables()` as `save_table_contract(..., "standard")`.

- [ ] **Step 7: Run schema/export regression tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_table_schema.py tests/test_table_export.py -q`

Expected: all selected tests pass, including existing standard rollback tests.

- [ ] **Step 8: Commit if implementation commits were explicitly authorized**

```powershell
git add deflector_tuning/table_schema.py deflector_tuning/table_export.py tests/test_table_schema.py tests/test_table_export.py
git commit -m "refactor: generalize transactional table contracts"
```

---

### Task 3: Y11/Z11 Runner, Manifest, and Figures

**Files:**
- Modify: `deflector_tuning/runner.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `deflector_tuning/visualization/admittance_sweep_plots.py`
- Modify: `tests/test_y_admittance_workflow.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_workflow_manifest.py`

**Interfaces:**
- Consumes: lane detection/loading/sampling from Task 1 and `save_table_contract()` from Task 2.
- Produces: standard/Y11/Z11 manifests with `table_contract`, direct Z11 plotting, and tables-only special-lane execution.

- [ ] **Step 1: Write failing manifest and runner tests**

```python
def test_y11_tables_only_uses_compact_contract_and_skips_plot(tmp_path: Path, monkeypatch) -> None:
    result = run_folder_analysis(..., tables_only=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(result.tables) == {"markers", "y11_pts"}
    assert result.tables["y11_pts"].name == "y11_pts.csv"
    assert result.figures == {}
    assert manifest["table_contract"] == "y11"
    assert manifest["table_schema_version"] == 2


def test_z11_normal_run_writes_table_and_complex_figure(tmp_path: Path) -> None:
    result = run_folder_analysis(sparameter_path=_z11_folder(tmp_path), ...)
    assert result.analysis_modes == ("z11_impedance",)
    assert set(result.tables) == {"markers", "z11_pts"}
    assert result.figures["z11_complex"]["marker_sweep"].exists()
```

- [ ] **Step 2: Run runner tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_y_admittance_workflow.py tests/test_runner.py tests/test_workflow_manifest.py -q`

Expected: Z11 routes into the unsupported simulation loader; Y11 tables-only raises; `table_contract` is absent.

- [ ] **Step 3: Extend manifest writing with a named contract**

Add `table_contract: str | None = None` to `write_manifest()`. Standard runs pass `"standard"`; direct lanes pass their lane. When set, serialize it alongside `table_schema_version` and `table_constants`.

- [ ] **Step 4: Implement one direct-lane runner**

Replace the Y-only helper with:

```python
def _run_one_port_matrix_analysis_from_runner(
    *, lane: Literal["y11", "z11"], tables_only: bool, ...
) -> RunResult:
    matrix = load_y11_touchstone_folder(...) if lane == "y11" else load_z11_touchstone_folder(...)
    points = sample_y11_markers(matrix, markers) if lane == "y11" else sample_z11_markers(matrix, markers)
    key = "y11_pts" if lane == "y11" else "z11_pts"
    save_result = save_table_contract(OrderedDict((("markers", markers), (key, points))), table_dir, lane)
```

If `tables_only`, reuse `cached_manifest_figures()` and do not call plotting functions. Otherwise render the parameter-specific complex sweep. Mixed Y/Z detection fails before creating output.

- [ ] **Step 5: Generalize the complex one-port plot with thin wrappers**

Keep `plot_y11_marker_sweep()` as a compatibility wrapper using `y_re_siemens`/`y_im_siemens` and millisiemens labels. Add `plot_z11_marker_sweep()` using `z_re_ohm`/`z_im_ohm` and ohm labels. Share only the private plotting loop.

- [ ] **Step 6: Run direct-lane and runner tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_y_admittance_workflow.py tests/test_one_port_matrix.py tests/test_runner.py tests/test_workflow_manifest.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit if implementation commits were explicitly authorized**

```powershell
git add deflector_tuning/runner.py deflector_tuning/workflows/manifest.py deflector_tuning/visualization/admittance_sweep_plots.py tests/test_y_admittance_workflow.py tests/test_one_port_matrix.py tests/test_runner.py tests/test_workflow_manifest.py
git commit -m "feat: run compact Y11 and Z11 analysis lanes"
```

---

### Task 4: Contract-Aware Batch Verification

**Files:**
- Modify: `run_all_folder_analyses.py`
- Modify: `tests/test_run_all_folder_analyses_cli.py`

**Interfaces:**
- Consumes: manifest `table_contract` and `TABLE_CONTRACTS`.
- Produces: `verify_table_contract_outputs(output_dir, manifest_path)` used for every tables-only batch success.

- [ ] **Step 1: Write failing batch verification tests**

```python
@pytest.mark.parametrize(
    ("contract", "key", "filename"),
    [("y11", "y11_pts", "y11_pts.csv"), ("z11", "z11_pts", "z11_pts.csv")],
)
def test_verify_table_contract_outputs_accepts_direct_lane(tmp_path: Path, contract: str, key: str, filename: str) -> None:
    manifest = _write_contract_output(tmp_path, contract, key, filename)
    verify_table_contract_outputs(tmp_path, manifest)


def test_verify_table_contract_outputs_rejects_wrong_declared_contract(tmp_path: Path) -> None:
    manifest = _write_contract_output(tmp_path, "y11", "z11_pts", "z11_pts.csv")
    with pytest.raises(ValueError, match="manifest table keys"):
        verify_table_contract_outputs(tmp_path, manifest)
```

- [ ] **Step 2: Run batch tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_run_all_folder_analyses_cli.py -q`

Expected: verifier assumes the standard 13-table set and rejects the direct contracts.

- [ ] **Step 3: Select verification rules from the manifest contract**

Implement `verify_table_contract_outputs()` to require schema version 2, a known `table_contract`, exact contract keys, canonical files, registered legacy absence, and forbidden-header absence. Keep `verify_standard_table_outputs()` as a compatibility wrapper if existing tests or callers require it.

- [ ] **Step 4: Run CLI/batch tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_run_all_folder_analyses_cli.py tests/test_run_folder_analysis_cli.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit if implementation commits were explicitly authorized**

```powershell
git add run_all_folder_analyses.py tests/test_run_all_folder_analyses_cli.py
git commit -m "feat: verify named table contracts in batch runs"
```

---

### Task 5: Repository and Real-Data Verification

**Files:**
- Verify: `deflector_tuning/**`, `run_folder_analysis.py`, `run_all_folder_analyses.py`, `tests/**`
- Regenerate: main-project `fig/analyses/sim_grid_260701_1DRcFine_Y`
- Regenerate: main-project `fig/analyses/sim_sweep_260713_coupler_tuner_zin`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified compact Y11/Z11 outputs in the existing analysis folders.

- [ ] **Step 1: Run the complete test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run isolated normal-mode smoke analyses**

Run Y11 and Z11 against temporary output folders under the visualization workspace. Assert the normal runners create `y11_pts.csv`/`z11_pts.csv` and the corresponding complex figure.

- [ ] **Step 3: Compare physical values before regeneration**

For Y11, compare the old `y_real_siemens`/`y_imag_siemens` values with new `y_re_siemens`/`y_im_siemens` by `source_file`, `marker_name`, and `freq_ghz`. For Z11, compare sampled `z_re_ohm`/`z_im_ohm` against the nearest raw `.z1p` values. Use `numpy.allclose(..., rtol=1e-10, atol=1e-12, equal_nan=True)`.

- [ ] **Step 4: Regenerate both real datasets in place**

```powershell
$main = 'C:\Users\9nugu\Documents\00_PARA\02_Area\TDC-HEM11-Kyhl-Iris-Center\Python_Deflector_tuning'
.\.venv\Scripts\python.exe run_all_folder_analyses.py `
  sim_grid_260701_1DRcFine_Y `
  sim_sweep_260713_coupler_tuner_zin `
  --tables-only --workers 1 `
  --data-root (Join-Path $main 'data') `
  --output-root (Join-Path $main 'fig\analyses')
```

Expected: both datasets succeed; the old Y11 file is removed only after validation.

- [ ] **Step 5: Audit regenerated contracts**

Check both manifests and table folders for schema version 2, the correct `table_contract`, `table_constants`, exact canonical keys/files, zero registered legacy files, compact headers, no staging residue, and valid figure paths. Confirm unrelated files remain.

- [ ] **Step 6: Run final verification**

```powershell
git diff --check
.\.venv\Scripts\python.exe -m pytest -q
git status --short
```

Expected: no whitespace errors, zero test failures, and only intended source/test/plan changes.

- [ ] **Step 7: Commit integration changes only if explicitly authorized**

```powershell
git add deflector_tuning/data_loading/one_port_matrix.py deflector_tuning/data_loading/readers/touchstone_reader.py deflector_tuning/data_loading/admittance.py deflector_tuning/table_schema.py deflector_tuning/table_export.py deflector_tuning/runner.py deflector_tuning/workflows/manifest.py deflector_tuning/visualization/admittance_sweep_plots.py run_all_folder_analyses.py tests/test_touchstone_reader.py tests/test_one_port_matrix.py tests/test_y_admittance_workflow.py tests/test_table_schema.py tests/test_table_export.py tests/test_runner.py tests/test_workflow_manifest.py tests/test_run_all_folder_analyses_cli.py docs/superpowers/plans/2026-07-13-y11-z11-compact-table-contract.md
git commit -m "feat: integrate compact Y11 and Z11 tables"
```
