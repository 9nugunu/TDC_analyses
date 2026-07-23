# Generic Touchstone VNA-50 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load CST S/Y/Z Touchstone files without mislabeling physical Y or Z values, then export one-port Z11 sweeps as VNA-equivalent 50-ohm S1P datasets.

**Architecture:** The low-level reader becomes parameter-neutral while retaining an `s_values` compatibility property. `FolderLoader` discovers one S/Y/Z family per folder and dispatches to thin long-form table adapters; `SimLoader` continues to attach CST comments and Navigator metadata. A new focused `vna50` module owns the pure impedance-to-reflection transform and the derived-dataset writer, with a small path-configurable CLI wrapper.

**Tech Stack:** Python 3.11+, pathlib, dataclasses, NumPy, pandas, pytest, Touchstone 1.x RI/DB parsing.

## Global Constraints

- Preserve the source folders `data/sim/sim_sweep_260713_coupler_tuner_insertion` and `data/sim/sim_sweep_260713_coupler_tuner_zin` unchanged.
- Write only the derived dataset `data/sim/sim_sweep_260713_coupler_tuner_vna50`.
- Apply `Gamma50 = (Z11 - 50) / (Z11 + 50)` independently at every frequency; do not use Y conversion or multiport load reduction.
- Do not add dataset-ID-specific conditionals.
- Preserve `data/sim`, `data/raw`, and `data/prepro` routing and simulation identity.
- Preserve existing S and direct-Y behavior; physical Y/Z values must not be normalized by the Touchstone `R` token.
- Reject mixed S/Y/Z folders and non-one-port VNA50 input explicitly.
- Do not clip `abs(Gamma50)` to one.
- Do not commit, stage, push, reset, delete, or overwrite unrelated user changes.

---

### Task 1: Parameter-neutral Touchstone reader

**Files:**
- Modify: `tests/test_touchstone_read.py`
- Modify: `tests/test_touchstone_header.py`
- Modify: `deflector_tuning/data_loading/readers/touchstone_reader.py`

**Interfaces:**
- Produces: `TouchstoneData.values: list[list[complex]]`.
- Preserves: `TouchstoneData.s_values` as a read-only compatibility property.
- Produces: `touchstone_parameter_from_suffix(path: str | Path) -> str` returning `S`, `Y`, or `Z`.

- [ ] **Step 1: Add failing Z1P and header-semantics tests**

```python
def test_read_z1p_ri_file_as_physical_impedance(tmp_path: Path) -> None:
    path = tmp_path / "case.z1p"
    path.write_text("# GHz Z RI R 1\n2.6 25 -10\n", encoding="utf-8")

    data = read_touchstone(path)

    assert data.header.parameter == "Z"
    assert data.header.reference_ohm == 1.0
    assert data.header.is_normalized is False
    assert data.values == [[25 - 10j]]


def test_reader_rejects_parameter_mismatch_between_suffix_and_header(tmp_path: Path) -> None:
    path = tmp_path / "case.z1p"
    path.write_text("# GHz S RI R 50\n2.6 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extension.*Z.*header.*S"):
        read_touchstone(path)
```

- [ ] **Step 2: Verify the new tests fail for the missing Z support**

Run: `conda run -n sys_env1 python -m pytest tests/test_touchstone_read.py tests/test_touchstone_header.py -q`

Expected: the Z1P test fails because `_value_count_from_suffix` accepts only S/Y and `TouchstoneData` has no neutral `values` field.

- [ ] **Step 3: Implement neutral values, S compatibility, and S/Y/Z suffix parsing**

```python
@dataclass(frozen=True)
class TouchstoneData:
    header: TouchstoneHeader
    frequency: list[float]
    values: list[list[complex]]

    @property
    def s_values(self) -> list[list[complex]]:
        """Compatibility view for existing S-parameter consumers."""
        return self.values


def touchstone_parameter_from_suffix(path: str | Path) -> str:
    match = re.fullmatch(r"\.(?P<parameter>[syz])(?P<ports>[1-9]\d*)p", Path(path).suffix.lower())
    if match is None:
        raise ValueError(f"Expected Touchstone extension like .s1p, .y1p, or .z1p; got {Path(path).name}")
    return match.group("parameter").upper()
```

Make `is_normalized` true only when `parameter == "S" and reference_ohm != 0.0`, and reject a suffix/header parameter mismatch after parsing the option line.

- [ ] **Step 4: Verify reader compatibility and Z support**

Run: `conda run -n sys_env1 python -m pytest tests/test_touchstone_read.py tests/test_touchstone_header.py -q`

Expected: all tests in both files pass.

---

### Task 2: Generic S/Y/Z folder discovery and adapters

**Files:**
- Modify: `tests/test_loader_touchstone.py`
- Modify: `tests/test_sim_metadata.py`
- Modify: `deflector_tuning/data_loading/loaders/folder_loader.py`
- Modify: `deflector_tuning/data_loading/loaders/sim_loader.py`

**Interfaces:**
- Consumes: `TouchstoneData.values` and `touchstone_parameter_from_suffix` from Task 1.
- Produces S columns unchanged: `s_name`, `s_real`, `s_imag`, `s_db`, `s_phase_deg`.
- Produces Y columns: `y_name`, `y_real_siemens`, `y_imag_siemens`.
- Produces Z columns: `z_name`, `z_real_ohm`, `z_imag_ohm`.

- [ ] **Step 1: Add failing discovery, mixed-family, and Z-adapter tests**

```python
def test_sim_loader_reads_z1p_with_navigator_metadata(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_zin"
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"tuner_insertion_depth"\n"1"\t"4"\n', encoding="utf-8"
    )
    (folder / "run_1.z1p").write_text(
        "# GHz Z RI R 1\n2.6 25 -10\n", encoding="utf-8"
    )

    table = DataLoader().load(folder)

    assert table.loc[0, "z_name"] == "Z11"
    assert table.loc[0, "z_real_ohm"] == pytest.approx(25.0)
    assert table.loc[0, "z_imag_ohm"] == pytest.approx(-10.0)
    assert table.loc[0, "run_id"] == 1
    assert table.loc[0, "sim_tuner_insertion_depth"] == pytest.approx(4.0)
    assert "s_real" not in table


def test_loader_rejects_mixed_touchstone_parameter_families(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_mixed"
    folder.mkdir(parents=True)
    (folder / "run_1.s1p").write_text("# GHz S RI R 50\n2.6 0 0\n", encoding="utf-8")
    (folder / "run_1.z1p").write_text("# GHz Z RI R 1\n2.6 50 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mixed Touchstone parameter families"):
        DataLoader().load(folder)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `conda run -n sys_env1 python -m pytest tests/test_loader_touchstone.py tests/test_sim_metadata.py -q`

Expected: `.z1p` is not discovered and mixed families are not rejected.

- [ ] **Step 3: Generalize discovery and row construction**

Use `touchstone_parameter_from_suffix` for discovery rather than a fixed extension set. In `load_touchstone`, compute the set of families before reading and raise if its length is not one. Build common metadata first, then add exactly one parameter-specific value block:

```python
if parameter == "S":
    row.update({"s_name": name, "s_real": value.real, "s_imag": value.imag,
                "s_db": _safe_db(value), "s_phase_deg": float(np.angle(value, deg=True)),
                "is_normalized": data.header.is_normalized})
elif parameter == "Y":
    row.update({"y_name": name, "y_real_siemens": value.real,
                "y_imag_siemens": value.imag})
elif parameter == "Z":
    row.update({"z_name": name, "z_real_ohm": value.real,
                "z_imag_ohm": value.imag})
```

Keep `reference_ohm` only as the parsed Touchstone header token/provenance for Y/Z; do not derive or scale physical values from it. Generate matrix names in Touchstone order using the active parameter letter.

- [ ] **Step 4: Generalize simulation run-ID parsing**

Change `_run_id_from_file_name` to accept `.[syz]Np` case-insensitively while leaving the existing numeric suffix behavior unchanged.

- [ ] **Step 5: Verify focused loader behavior**

Run: `conda run -n sys_env1 python -m pytest tests/test_loader_touchstone.py tests/test_sim_metadata.py -q`

Expected: all focused tests pass, including pre-existing S loader tests.

---

### Task 3: Route the existing direct-Y workflow through the generic loader

**Files:**
- Modify: `tests/test_y_admittance_workflow.py`
- Modify: `deflector_tuning/data_loading/admittance.py`

**Interfaces:**
- Consumes: the generic Y table from `SimLoader.load_touchstone`.
- Preserves: `load_y11_touchstone_folder(path) -> pd.DataFrame` and its current public columns/order.

- [ ] **Step 1: Strengthen the existing Y test as a compatibility guard**

Add assertions that the direct-Y table contains no `s_real`/`is_normalized` columns and that its values remain exactly the physical values written in the Y1P file.

- [ ] **Step 2: Run the strengthened test before refactoring**

Run: `conda run -n sys_env1 python -m pytest tests/test_y_admittance_workflow.py::test_load_y11_touchstone_folder_preserves_cst_values_and_result_navigator_metadata -q`

Expected: the new assertion about `is_normalized` fails against the current dedicated loader, establishing the intended generic-adapter boundary.

- [ ] **Step 3: Replace duplicate Y parsing with the generic SimLoader path**

```python
table = SimLoader().load_touchstone(folder)
if "y_name" not in table:
    raise ValueError(f"Expected Y-parameter Touchstone data in {folder}")
if set(table["y_name"].dropna()) != {"Y11"}:
    raise ValueError(f"Expected one-port Y11 data in {folder}")
```

Retain marker extraction and sampling functions unchanged. Remove only the now-duplicated result-Navigator, run-ID, unit-conversion, and Y-file loops.

- [ ] **Step 4: Verify the complete direct-Y workflow**

Run: `conda run -n sys_env1 python -m pytest tests/test_y_admittance_workflow.py -q`

Expected: all direct-Y loading, marker, and runner tests pass.

---

### Task 4: Pure Zin-to-reflection transform

**Files:**
- Create: `tests/test_vna50_transform.py`
- Create: `deflector_tuning/data_loading/vna50.py`

**Interfaces:**
- Produces: `impedance_to_reflection(z_input: object, reference_ohm: float = 50.0) -> np.ndarray`.

- [ ] **Step 1: Write formula and validation tests**

```python
def test_impedance_to_reflection_uses_fixed_reference_impedance() -> None:
    values = np.array([50 + 0j, 0 + 0j, 100 + 50j])
    expected = (values - 50.0) / (values + 50.0)
    np.testing.assert_allclose(impedance_to_reflection(values), expected)


@pytest.mark.parametrize("reference", [0.0, -50.0, np.nan, np.inf])
def test_impedance_to_reflection_rejects_invalid_reference(reference: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        impedance_to_reflection([50 + 0j], reference)


def test_impedance_to_reflection_rejects_nonfinite_z() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        impedance_to_reflection([complex(np.nan, 0.0)])


def test_impedance_to_reflection_rejects_singular_denominator() -> None:
    with pytest.raises(ValueError, match="singular"):
        impedance_to_reflection([-50 + 0j])
```

- [ ] **Step 2: Run the new tests and confirm import failure**

Run: `conda run -n sys_env1 python -m pytest tests/test_vna50_transform.py -q`

Expected: collection fails because `deflector_tuning.data_loading.vna50` does not exist.

- [ ] **Step 3: Implement the minimal pure transform**

```python
def impedance_to_reflection(z_input: object, reference_ohm: float = 50.0) -> np.ndarray:
    z0 = float(reference_ohm)
    if not np.isfinite(z0) or z0 <= 0.0:
        raise ValueError("reference_ohm must be positive and finite")
    impedance = np.asarray(z_input, dtype=complex)
    if not np.all(np.isfinite(impedance.real) & np.isfinite(impedance.imag)):
        raise ValueError("Z input contains non-finite values")
    denominator = impedance + z0
    singular_tolerance = np.finfo(float).eps * max(1.0, z0) * 16.0
    if np.any(np.abs(denominator) <= singular_tolerance):
        raise ValueError("Z input produces a numerically singular denominator")
    return (impedance - z0) / denominator
```

- [ ] **Step 4: Verify the transform tests pass**

Run: `conda run -n sys_env1 python -m pytest tests/test_vna50_transform.py -q`

Expected: all pure-transform tests pass.

---

### Task 5: Derived S1P dataset writer and CLI

**Files:**
- Modify: `tests/test_vna50_transform.py`
- Modify: `deflector_tuning/data_loading/vna50.py`
- Create: `scripts/export_vna50_touchstone.py`

**Interfaces:**
- Produces: immutable `Vna50Export(dataset_path, touchstone_files, manifest_path)`.
- Produces: `export_vna50_touchstone_dataset(source_path, output_path, reference_ohm=50.0) -> Vna50Export`.
- CLI: `python scripts/export_vna50_touchstone.py SOURCE OUTPUT [--reference-ohm 50]`.

- [ ] **Step 1: Add writer round-trip, conflict, and grid-validation tests**

Create a temporary `sim_sweep_260713_tuner_zin` with two Z1P files and Navigator rows. Export it, load the output through `DataLoader`, and assert:

```python
assert export.dataset_path == output
assert [path.name for path in export.touchstone_files] == ["run_1.s1p", "run_2.s1p"]
assert set(reloaded["s_name"]) == {"S11"}
assert reloaded["reference_ohm"].unique().tolist() == [50.0]
np.testing.assert_allclose(
    reloaded["s_real"].to_numpy() + 1j * reloaded["s_imag"].to_numpy(),
    (source_z - 50.0) / (source_z + 50.0),
)
assert (output / "result_navigator.csv").read_bytes() == (source / "result_navigator.csv").read_bytes()
```

Also assert that mismatched per-file frequency grids, missing/duplicated file run IDs, non-Z11 input, and an existing output without an owned matching manifest raise targeted `ValueError`s.

- [ ] **Step 2: Run writer tests and confirm RED**

Run: `conda run -n sys_env1 python -m pytest tests/test_vna50_transform.py -q`

Expected: writer tests fail because the export dataclass/function are absent.

- [ ] **Step 3: Implement source-table validation and deterministic S1P writing**

Load via `DataLoader`, require exactly `Z11`, assemble complex Z from `z_real_ohm`/`z_imag_ohm`, and verify one run ID per source file plus identical frequency arrays. Write each file as:

```text
! Derived from CST Z11
! Gamma50 = (Zin - 50) / (Zin + 50)
! Source dataset: <source dataset id>
# GHz S RI R 50
<freq_ghz> <gamma_real> <gamma_imag>
```

Use enough significant digits (`.17g`) for numerical round trips. Copy `result_navigator.csv` byte-for-byte. Validate the output dataset ID with `DataLayer.SIM` before creating it.

- [ ] **Step 4: Implement owned rerun/conflict rules and manifest**

Write `transform_manifest.json` with:

```python
{
    "transform": "z11_to_vna_reflection",
    "source_dataset": source.name,
    "output_dataset": output.name,
    "formula": "Gamma = (Zin - Z0) / (Zin + Z0)",
    "reference_ohm": reference_ohm,
    "input_parameter": "Z11",
    "output_parameter": "S11",
    "files": file_records,
    "frequency_min_ghz": frequency[0],
    "frequency_max_ghz": frequency[-1],
    "point_count_per_file": len(frequency),
    "maximum_abs_gamma": maximum_abs_gamma,
}
```

An existing output is writable only when its manifest matches the transform, source dataset, output dataset, reference impedance, and exact source/output file mapping. Reject extra S1P files. Never delete output files as part of rerun handling.

- [ ] **Step 5: Add the path-configurable CLI wrapper**

```python
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--reference-ohm", type=float, default=50.0)
result = export_vna50_touchstone_dataset(args.source, args.output, args.reference_ohm)
print(result.dataset_path)
```

- [ ] **Step 6: Verify writer round-trip and error handling**

Run: `conda run -n sys_env1 python -m pytest tests/test_vna50_transform.py tests/test_loader_touchstone.py tests/test_sim_metadata.py tests/test_y_admittance_workflow.py -q`

Expected: all new and compatibility tests pass.

---

### Task 6: Production export and full verification

**Files:**
- Create by running the workflow: `data/sim/sim_sweep_260713_coupler_tuner_vna50/`
- No source-code edits unless a failing verification receives a new regression test first.

**Interfaces:**
- Consumes: `data/sim/sim_sweep_260713_coupler_tuner_zin/*.z1p`.
- Produces: four `.s1p` files, copied `result_navigator.csv`, and `transform_manifest.json`.

- [ ] **Step 1: Run the production export**

Run:

```powershell
conda run -n sys_env1 python scripts/export_vna50_touchstone.py `
  data/sim/sim_sweep_260713_coupler_tuner_zin `
  data/sim/sim_sweep_260713_coupler_tuner_vna50
```

Expected: the output path is printed and no source file is modified.

- [ ] **Step 2: Numerically audit all four files**

Run a read-only Python check that loads source Z and output S, joins by source stem and frequency, recomputes `(Z11 - 50)/(Z11 + 50)`, and requires maximum complex absolute error below `1e-14`. Require four source/output files, 10,001 points per file, 2.6–3.0 GHz, `R 50`, and finite complex values.

- [ ] **Step 3: Compare an existing CST 50-ohm artifact where file pairing is available**

Use one matching `sim_grid_260701_1DRcFine_Y`/`sim_grid_260701_1DRcFine_50norm` run. Compute `Z11 = 1 / Y11`, then `Gamma50`, and report the maximum complex difference against CST `50norm`. This is a read-only cross-check; do not alter the transformation to force agreement if the CST export convention differs.

- [ ] **Step 4: Run the complete relevant test suite**

Run:

```powershell
conda run -n sys_env1 python -m pytest `
  tests/test_touchstone_read.py `
  tests/test_touchstone_header.py `
  tests/test_loader_touchstone.py `
  tests/test_sim_metadata.py `
  tests/test_y_admittance_workflow.py `
  tests/test_vna50_transform.py -q
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Inspect scope and whitespace**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors; only the approved design/plan, generic loader/transform code, focused tests, CLI, and generated VNA50 dataset are attributable to this task. Pre-existing user changes remain present and untouched except for the narrowly required overlaps in `sim_loader.py`, `admittance.py`, and their tests.
