# Compact Analysis Table Schema Design

## Goal

Replace the verbose, duplicate-heavy standard analysis-table schema with one concise canonical schema used by analysis code, plots, tests, scripts, manifests, and CSV files. Regenerate the existing batch-managed tables in place without deleting unrelated user or special-purpose files.

## Scope

Phase 1 covers the 13 standard tables produced by `build_marker_analysis()` and saved under `fig/analyses/<dataset_id>/tables`:

1. `markers`
2. `marker_points`
3. `marker_phase_polar`
4. `kyhl_f2pi3_normalized_admittance_audit`
5. `kyhl_admittance_points`
6. `kyhl_admittance_transitions`
7. `cell_iris_response_comparison`
8. `coupler_cavity_parameter_estimates`
9. `grid_rc_line_scan`
10. `phase_advance`
11. `phase_summary`
12. `nodal_shift`
13. `geometry_phase_response`

Phase 1 also updates every in-repository producer and consumer of those tables, including analysis modules, visualizations, runner mode detection, tests, `scripts/build_revised_kyhl_validation_metrics.py`, `run_folder_analysis.py`, and `run_all_folder_analyses.py`.

The following are out of scope for Phase 1:

- special Y11 table schemas;
- CST dispersion-only table schemas;
- CST profile-only table schemas;
- manually curated tables under `outputs/*`;
- renaming source dataset folders or source files;
- rewriting historical results outside `fig/analyses/*/tables`.

These outputs may be refreshed by the batch runner when their dataset category is encountered, but their schemas and filenames are not renamed or cleaned in Phase 1.

## Evidence and Constraints

An audit of the existing batch results found 54 `tables` directories and 409 CSV files outside `.omx`. The standard tables reached 74 columns in the KYHL audit, 77 columns in KYHL transitions, 68 columns in coupler estimates, and 58 columns in marker points.

The audit also distinguished semantic aliases from coincidentally equal values:

- Semantic aliases that are equal in every populated historical result include several KYHL admittance aliases such as `raw_pseudo_admittance_real`, `admittance_real`, `operation_scaled_admittance_real`, and `kyhl_operation_real`.
- `from_freq_ghz` and `to_freq_ghz` are equal throughout transition tables because a transition compares positions at one marker frequency.
- Source metadata such as `sim_L_c` and `sim_d`, or `sim_coupler_path_h` and `sim_r_c`, is equal only in some datasets. These remain separate because they represent distinct CST parameters and may diverge in future inputs.

The design must therefore be semantic and deterministic. It must never remove a column merely because two value vectors happen to match in one dataset.

## Canonical Schema Boundary

The canonical concise names originate in analysis producers. There is no old-name internal DataFrame followed by a rename-only CSV presentation layer.

Existing concise shared columns remain unchanged, including:

- `dataset_id`
- `source_file`
- `run_id`
- `marker_name`
- `freq_ghz`
- `s_db`
- `s_phase_deg`
- `tune_position`
- concise simulation coordinates such as `sim_r_c` and `sim_w_c`

Verbose derived quantities are renamed at their producer. Every plot, mode detector, test, and script consumes the same concise name. A removed semantic alias is not recreated for compatibility.

Source metadata is handled conservatively:

- distinct source parameters are not merged based on equal current values;
- known verbose source fields may receive a global column alias independent of dataset identity;
- unknown `sim_*` fields remain unchanged rather than receiving a lossy automatic abbreviation;
- dataset-specific `if dataset_id == ...` schema branches are forbidden.

Initial global source aliases are:

| Source column | Canonical column |
|---|---|
| `sim_tuner_insertion_depth` | `sim_tuner_depth` |
| `sim_coupler_path_bot2_width` | `sim_cpl_bot2_w` |

## Naming Vocabulary

Names use lower-case snake case. A physical quantity remains recognizable; abbreviations shorten qualifiers and components rather than replacing the physical quantity with a one-letter symbol.

| Concept | Canonical token |
|---|---|
| admittance | `admit` |
| real component | `re` |
| imaginary component | `im` |
| magnitude | `mag` |
| angle | `ang` |
| operation coordinate | `op` |
| reference coordinate | `ref` |
| normalized transmission-line coordinate | `line` |
| operating-mode-normalized coordinate | `mode` |
| physical value | `phys` |
| error | `err` |
| external quality factor | `q_ext` |
| comparison | `cmp` |
| parameters | `params` |
| points | `pts` |
| transitions | `steps` |
| statistics | `stats` |

Known units remain explicit suffixes such as `_deg`, `_ghz`, `_ohm`, and `_siemens`. Dimensionless values do not gain an artificial unit suffix.

Representative canonical mappings are:

| Old column | Canonical column |
|---|---|
| `operation_scaled_admittance_real` | `op_admit_re` |
| `operation_scaled_admittance_imag` | `op_admit_im` |
| `operation_scaled_admittance_abs` | `op_admit_mag` |
| `operation_scaled_admittance_angle_deg` | `op_admit_ang_deg` |
| `line_normalized_admittance_real` | `line_admit_re` |
| `mode_normalized_admittance_angle_deg` | `mode_admit_ang_deg` |
| `abs_operation_axis_error_deg` | `op_admit_axis_err_deg` |
| `phase_advance_0to360_deg` | `phase_adv_deg` |
| `phase_error_from_target_deg` | `phase_err_deg` |
| `abs_phase_error_from_target_deg` | `phase_err_abs_deg` |
| `target_external_quality_factor` | `q_ext_target` |

The implementation maintains one explicit migration map used by schema contract tests. Automatic word replacement is not used because it can create collisions or obscure scientific meaning.

## Table Keys and Filenames

Internal `AnalysisTables` keys, manifest table keys, and CSV stems use the same canonical name:

| Legacy key / filename | Canonical key / filename |
|---|---|
| `markers` / `markers.csv` | `markers` / `markers.csv` |
| `marker_points` / `marker_points.csv` | `marker_pts` / `marker_pts.csv` |
| `marker_phase_polar` / `marker_phase_polar.csv` | `phase_polar` / `phase_polar.csv` |
| `kyhl_f2pi3_normalized_admittance_audit` / `kyhl_f2pi3_normalized_admittance_audit.csv` | `kyhl_admit_audit` / `kyhl_admit_audit.csv` |
| `kyhl_admittance_points` / `kyhl_admittance_points.csv` | `kyhl_admit_pts` / `kyhl_admit_pts.csv` |
| `kyhl_admittance_transitions` / `kyhl_admittance_transitions.csv` | `kyhl_admit_steps` / `kyhl_admit_steps.csv` |
| `cell_iris_response_comparison` / `cell_iris_response_comparison.csv` | `cell_iris_cmp` / `cell_iris_cmp.csv` |
| `coupler_cavity_parameter_estimates` / `coupler_cavity_parameter_estimates.csv` | `coupler_params` / `coupler_params.csv` |
| `grid_rc_line_scan` / `grid_rc_line_scan.csv` | `rc_line` / `rc_line.csv` |
| `phase_advance` / `phase_advance.csv` | `phase_adv` / `phase_adv.csv` |
| `phase_summary` / `phase_summary.csv` | `phase_stats` / `phase_stats.csv` |
| `nodal_shift` / `nodal_shift.csv` | `nodal_shift` / `nodal_shift.csv` |
| `geometry_phase_response` / `geometry_phase_response.csv` | `geom_phase` / `geom_phase.csv` |

Figure group names may remain domain-oriented when changing them would not improve table clarity, but every lookup into `AnalysisTables` uses the canonical table key.

## Duplicate Removal Rules

Duplicate removal happens at the producer and is defined explicitly per analysis family.

Rules include:

- KYHL point and transition tables retain the complete operation-coordinate columns (`op_admit_re`, `op_admit_im`, `op_admit_mag`, and `op_admit_ang_deg`) and remove the parallel `raw_pseudo_*`, generic `admittance_*`, and `kyhl_operation_*` columns; the raw reflection remains available as canonical `gamma_*` columns;
- the dedicated KYHL normalization audit retains physically distinct line-normalized, mode-normalized, and physical admittance coordinates because comparing those coordinates is the purpose of that table;
- the normalization audit removes generic `admittance_*` aliases of line admittance and, because positive mode normalization leaves the angle unchanged, retains only `mode_admit_ang_deg` while omitting the duplicate line-admittance angle;
- use one `freq_ghz` column for a same-marker transition instead of equal `from_freq_ghz` and `to_freq_ghz` columns;
- retain both start and end values for quantities that can differ across a transition;
- do not deduplicate all-null columns by pairwise value comparison;
- do not deduplicate `sim_*` parameters based on current values.

Schema contract tests list forbidden legacy aliases and confirm that each canonical quantity appears once.

## CSV Projection and Manifest Context

Internal DataFrames retain columns required by analysis and plotting. CSVs use the same canonical names but project out non-result metadata that does not vary within that table.

Each table export specification classifies columns into:

- required identity columns;
- result columns, which are always written even if their values are constant;
- metadata columns, which are written only when they vary or are explicitly required for row identity.

For a populated table:

- `dataset_id` is always retained;
- required row identifiers are retained;
- result columns are retained in a fixed documented order;
- varying sweep and metadata columns are retained;
- all-null metadata columns are omitted;
- single-valued metadata columns are omitted from the CSV and recorded in the dataset manifest.

For an empty table, the CSV contains the canonical required identity and result headers but no propagated metadata headers. This prevents empty outputs from carrying dozens of irrelevant columns.

The manifest gains:

```json
{
  "table_schema_version": 2,
  "table_constants": {
    "marker_pts": {
      "marker_role": "sim",
      "reference_ohm": 50.0
    }
  }
}
```

`table_constants` is keyed by canonical table key. Values must be JSON-safe native scalars or `null`. Result columns are never moved to `table_constants`, even when constant. `table_schema_version: 2` applies to the standard marker-analysis lane; Phase 1 special lanes retain their existing schema identity.

## Safe Save and Legacy Cleanup

New tables are written to a staging location in the target `tables` directory. Existing managed CSVs remain untouched until every new standard table has been serialized successfully.

The commit sequence is:

1. serialize all 13 canonical CSVs to staging files;
2. validate filenames and headers against the schema registry;
3. replace canonical managed files;
4. remove only legacy standard filenames whose canonical filename differs;
5. leave unrelated CSVs and subdirectories untouched;
6. remove staging artifacts.

If staging or validation fails, the existing results remain unchanged. If replacement fails, the saver restores backed-up managed files before reporting the error. Legacy cleanup never runs before canonical replacement succeeds.

The unchanged filenames `markers.csv` and `nodal_shift.csv` are replaced as managed canonical files; they are not treated as stale legacy files.

## Runner Changes

Both one-dataset and batch CLIs gain `--tables-only`.

With `--tables-only`:

- loaders and standard analyses run normally;
- canonical tables and table constants are regenerated;
- figure rendering is skipped;
- existing figure files are not deleted;
- existing manifest figure paths are preserved only when the referenced files still exist;
- the manifest table paths, table schema version, analysis modes, and table constants are refreshed;
- special Y11, dispersion, and profile lanes regenerate their existing table products without Phase 1 renaming and skip their figures where the lane supports figure suppression.

`run_all_folder_analyses.py` passes the flag to every `BatchTask`. Successful standard marker tasks verify that canonical managed filenames exist and that differing legacy standard filenames no longer exist. A dataset failure remains isolated and is reported in `failed_datasets` using the existing keep-going batch behavior.

Dataset naming validation remains strict. The compact schema change must not introduce silent skipping or dataset-name-specific routing.

## Existing Result Regeneration

Regeneration targets batch-managed results under `fig/analyses/<dataset_id>/tables`. Manually curated `outputs/*` tables are not rewritten in Phase 1.

The execution order is:

1. run the focused unit and integration tests;
2. regenerate one representative sweep dataset;
3. regenerate one representative grid dataset;
4. compare row counts and scientific values with the pre-migration outputs using the explicit migration map;
5. run the batch table-only refresh for valid, batch-managed datasets;
6. audit all refreshed standard marker-analysis directories for canonical filenames, forbidden legacy filenames, forbidden legacy columns, and manifest schema version 2.

Historical CSVs are replaced in their existing dataset directories. No parallel `v2` directory is created.

## Error Handling

- A canonical-name collision raises an error before any managed file is replaced.
- A missing required result column raises an error naming the table and column.
- A non-JSON-safe manifest constant is converted through an explicit scalar normalizer or rejected with table and column context.
- A stale legacy file is removed only when it belongs to the fixed legacy filename registry.
- A failed dataset does not trigger cleanup in that dataset directory.
- Batch completion returns nonzero when any dataset fails and prints the existing failure summary.

## Testing

Implementation follows test-driven development.

Required tests cover:

- every renamed `AnalysisTables` key;
- every canonical CSV filename;
- representative mappings from each affected analysis module;
- explicit removal of semantic aliases;
- preservation of distinct source metadata even when sample values match;
- canonical admittance column names containing `admit`, never one-letter `y` replacements;
- preservation of unit suffixes;
- constant metadata projection into `manifest.json`;
- retention of constant result columns in CSVs;
- compact headers for empty tables;
- legacy cleanup after successful replacement;
- rollback or preservation after injected serialization and replacement failures;
- unrelated CSV preservation;
- one-dataset and batch `--tables-only` behavior;
- preservation of existing valid figure paths in tables-only manifests;
- compatibility of `scripts/build_revised_kyhl_validation_metrics.py` with canonical columns;
- representative sweep and grid end-to-end outputs;
- the full repository test suite.

## Success Criteria

The work is complete when:

- all Phase 1 producers and consumers use canonical concise names directly;
- no forbidden legacy alias remains in executable code or refreshed standard CSV headers;
- all 13 standard table keys and filenames match the canonical registry;
- admittance-derived columns retain the `admit` token and scientific coordinate distinctions;
- refreshed CSVs contain result columns, row identifiers, and varying sweep columns without repeated constant metadata;
- manifest schema version 2 records omitted constant metadata;
- refreshed dataset directories contain no differing legacy standard filename;
- unrelated and special-purpose files are preserved;
- focused tests and the full suite pass;
- representative value comparisons show that renaming and deduplication did not change scientific results.
