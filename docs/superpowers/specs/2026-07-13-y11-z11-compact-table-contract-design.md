# Y11/Z11 Compact Table Contract Design

## Goal

Bring the existing direct-Y11 workflow and the currently unsupported direct-Z11 data into the same compact, transactional CSV persistence system used by the standard marker-analysis tables. Scope is limited to one-port `.y1p` and `.z1p` inputs, so the represented matrix elements are Y11 and Z11 only.

## Naming Contract

Direct Y11 and Z11 use matrix-symbol names, as selected by the user:

| Lane | Table key | Canonical filename | Legacy filename |
| --- | --- | --- | --- |
| Y11 markers | `markers` | `markers.csv` | `markers.csv` |
| Y11 samples | `y11_pts` | `y11_pts.csv` | `y11_marker_points.csv` |
| Z11 markers | `markers` | `markers.csv` | `markers.csv` |
| Z11 samples | `z11_pts` | `z11_pts.csv` | none |

Shared sample columns remain consistent with the standard compact schema: `dataset_id`, `source_file`, `run_id`, `marker_name`, `marker_role`, `marker_source`, `freq_target_ghz`, `freq_ghz`, and `freq_error_ghz`.

The direct physical values are:

- Y11: `y_re_siemens`, `y_im_siemens`
- Z11: `z_re_ohm`, `z_im_ohm`

The constant element labels (`Y11` or `Z11`), source format, reference value, and other constant context are recorded under `table_constants` in the manifest. Varying simulation parameters such as `sim_r_c` remain in the CSV.

## Loading and Sampling

The Touchstone reader accepts `.zNp` in addition to its existing `.sNp` and `.yNp` suffixes. A direct one-port matrix loader validates that each file:

1. has the expected `Y` or `Z` Touchstone parameter,
2. contains exactly one complex matrix value per frequency, and
3. belongs to a folder containing only the selected direct lane for routing purposes.

Y11 and Z11 share the same result-navigator metadata parsing and nearest-marker sampling algorithm. Thin Y11/Z11 wrappers expose parameter-specific column names without mixing Siemens and ohms in a generic value column. A folder containing both `.y1p` and `.z1p` files is rejected as ambiguous instead of selecting one silently.

## Persistence Architecture

The central table schema registry gains separate ordered contracts for:

- the existing 13-table standard marker lane,
- the two-table direct Y11 lane, and
- the two-table direct Z11 lane.

The transactional saver is generalized to accept one explicit contract. It stages every table required by that contract, validates key order and headers, backs up canonical and registered legacy files, replaces the complete canonical set, removes only registered differing legacy files, and rolls back on failure. The existing `save_standard_tables()` remains as a compatibility wrapper around the standard contract.

Manifests declare `table_schema_version: 2`, `table_contract: standard|y11|z11`, and `table_constants`. Batch verification selects the expected filenames from `table_contract`; it does not assume every successful tables-only task has 13 files.

## Runner Behavior

Direct `.y1p` and `.z1p` folders are routed before the standard S-parameter marker pipeline.

- Normal Y11 execution writes `markers.csv` and `y11_pts.csv`, then renders the existing complex-Y11 sweep figure using the compact Y columns.
- Normal Z11 execution writes `markers.csv` and `z11_pts.csv`, then renders a symmetric complex-Z11 sweep figure.
- `--tables-only` works for both lanes, skips plot rendering, and preserves only existing figure paths in the manifest.
- Standard S-parameter behavior and its 13-table contract remain unchanged.

The existing Y11 dataset is regenerated in place. Its registered legacy `y11_marker_points.csv` is removed only after the new pair is saved successfully. The current Z11 dataset is processed for the first time from its `.z1p` inputs.

## Errors and Safety

- Missing `.y1p`/`.z1p` files produce a clear lane-specific error.
- Header/extension parameter mismatches fail before output replacement.
- Multi-port Y/Z inputs are outside scope and fail explicitly.
- Ambiguous mixed Y/Z folders fail explicitly.
- Failed datasets retain their previous managed files and unrelated files.
- Special profile and dispersion workflows remain outside these contracts.

## Verification

Tests cover suffix parsing, Y11 and Z11 loading, compact sampling columns, ambiguous routing, transactional legacy cleanup and rollback, normal and tables-only runner behavior, manifest contracts, and batch verification. Final validation includes the full test suite plus real regeneration/audit of:

- `sim_grid_260701_1DRcFine_Y`
- `sim_sweep_260713_coupler_tuner_zin`

Each regenerated directory must have its exact two canonical tables, no registered differing legacy file, schema version 2, the correct table contract, compact headers, and valid preserved/generated figure paths.
