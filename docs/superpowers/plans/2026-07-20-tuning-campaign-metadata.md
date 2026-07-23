# Tuning Campaign Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compact YAML-backed tuning states and issues that are automatically recognized by the existing folder-analysis command without rewriting any TOML file.

**Architecture:** A focused `tuning_campaign` module owns immutable models, YAML parsing, validation, campaign lookup, and normalized table conversion. Existing CLI entrypoints keep their current arguments and call one post-analysis hook; unmatched ordinary datasets remain unchanged. Campaign identity is added to the existing manifest so later experiment-versus-simulation plots can consume the same source of truth without dataset-specific branches.

**Tech Stack:** Python 3.11, dataclasses, pathlib, PyYAML, pandas, pytest.

## Global Constraints

- Preserve centralized `data/sim`, `data/raw`, and `data/prepro` routing.
- Do not add branches that test for a specific dataset name.
- `config/project_defaults.toml` remains unchanged and is not duplicated.
- State IDs encode sequence only; Torque 0 `s000` is the unique RF baseline.
- Missing or uncertain mechanical/RF facts remain pending, provisional, or explicit verification targets.
- Do not commit, push, reset, delete, or stage files unless the user explicitly asks.

---

### Task 1: Typed YAML loader and validation

**Files:**
- Create: `deflector_tuning/tuning_campaign.py`
- Modify: `pyproject.toml`
- Test: `tests/test_tuning_campaign.py`

**Interfaces:**
- Produces: `TuningCampaign`, `TuningState`, `TuningIssue`, `load_tuning_campaign(path, *, data_root=None)`, `find_tuning_campaign(dataset_id, campaign_dir, *, data_root=None)`.
- `TuningCampaign.states` and `.issues` preserve YAML insertion order as mappings.
- Issue defaults are `stat="open"`, `sev="warn"`, and `action="flag"`.

- [ ] **Step 1: Write failing compact-load tests**

Create a temporary YAML containing `s000`, `s001`, pending `s002`, and compact `i001`. Assert typed values, expanded defaults, and `check == ("output",)`.

- [ ] **Step 2: Run the focused test and verify the module is missing**

Run: `python -m pytest tests/test_tuning_campaign.py -q`

Expected: collection fails because `deflector_tuning.tuning_campaign` does not exist.

- [ ] **Step 3: Implement immutable models and compact YAML parsing**

Implement frozen dataclasses. Reject non-mapping roots and unsupported `schema`. Parse optional numeric fields without converting booleans to numbers. Keep concise YAML keys at the file boundary and expose readable Python attributes.

- [ ] **Step 4: Add validation tests**

Cover unique baseline, missing `prev`/`after`/`verify` references, cyclic `prev`, completed state without `data`, duplicate completed datasets, missing simulation/raw directories when `data_root` is supplied, and pending state without `data`.

- [ ] **Step 5: Implement validation and lookup**

`find_tuning_campaign` loads all `*.yaml` and `*.yml` files, returns `None` for no match, and raises on multiple matches. A campaign matches its simulation ID or any state dataset ID. Dataset existence is checked only when `data_root` is supplied.

- [ ] **Step 6: Declare and verify the YAML dependency**

Add `PyYAML` to `pyproject.toml` dependencies. Verify the active environment can import `yaml`; install the editable project only if required for the test environment.

- [ ] **Step 7: Run Task 1 tests**

Run: `python -m pytest tests/test_tuning_campaign.py -q`

Expected: all tests pass.

---

### Task 2: Normalized metadata and manifest registration

**Files:**
- Create: `deflector_tuning/workflows/tuning_campaign.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Test: `tests/test_tuning_campaign_workflow.py`
- Test: `tests/test_workflow_manifest.py`

**Interfaces:**
- Consumes: `TuningCampaign` from Task 1.
- Produces: `campaign_state_table(campaign) -> pandas.DataFrame`, `campaign_issue_table(campaign) -> pandas.DataFrame`, `export_campaign_tables(campaign, output_dir)`, and `register_tuning_campaign(manifest_path, *, config_path, campaign, state_id)`.

- [ ] **Step 1: Write failing normalized-table tests**

Assert one row per state/issue, explicit effective issue defaults, a blank dataset for pending `s002`, and no invented output-coupler impact value. `check` is exported as a stable semicolon-separated string.

- [ ] **Step 2: Implement normalized table conversion and optional export**

Keep the YAML file as source of truth. Conversion is in-memory unless
`export_campaign_tables` is explicitly called with a derived `data/prepro`
output directory. Test that it writes `states.csv` and `issues.csv` without
modifying the YAML source. No CSV is written by ordinary folder analysis.

- [ ] **Step 3: Write failing manifest-registration tests**

Start from an existing manifest, register campaign metadata, and assert preservation of all previous fields plus:

```json
{
  "tuning_campaign": {
    "campaign_id": "iris_260701",
    "config_path": "config/tuning_campaigns/iris_260701.yaml",
    "state_id": "s001",
    "state_status": "done",
    "issue_ids": ["i001"]
  }
}
```

- [ ] **Step 4: Implement atomic manifest registration**

Reuse the manifest module's temporary-file replacement pattern. Link an issue to the analyzed state when its `after` or `verify` field equals that state. Preserve unrelated manifest content.

- [ ] **Step 5: Run Task 2 tests**

Run: `python -m pytest tests/test_tuning_campaign_workflow.py tests/test_workflow_manifest.py -q`

Expected: all tests pass.

---

### Task 3: Existing-command auto-discovery and current campaign file

**Files:**
- Create: `config/tuning_campaigns/iris_260701.yaml`
- Modify: `run_folder_analysis.py`
- Modify: `run_all_folder_analyses.py`
- Modify: `tests/test_run_folder_analysis_cli.py`
- Modify: `tests/test_run_all_folder_analyses_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `find_tuning_campaign` and `register_tuning_campaign` from Tasks 1-2.
- Existing CLI syntax remains unchanged.

- [ ] **Step 1: Write failing single-run integration test**

Mock the standard folder analysis result and campaign lookup. Call `main()` with the existing positional dataset syntax and assert that matching campaign/state metadata is registered after normal analysis. Assert that no-match performs no registration and still returns zero.

- [ ] **Step 2: Implement a small post-analysis hook**

After `run_folder_analysis` returns, derive the dataset ID through the existing centralized naming path, look in `config/tuning_campaigns`, and register a unique match. Do not add required CLI arguments and do not read or alter `project_defaults.toml`.

- [ ] **Step 3: Write and implement batch-run coverage**

Add campaign-directory information to `BatchTask` with a repository default and call the same post-analysis helper after each completed dataset. Confirm multiprocessing tasks remain pickleable and ordinary datasets remain unchanged.

- [ ] **Step 4: Add the current campaign source file**

Record `s000` as Torque 0 baseline, `s001` as the completed Torque 13.5 provisional state, `s002` as pending after adding only the tuner bolt, and `i001` as the input-coupler lower-right tuner disconnect with output-coupler verification at `s002`. Record port extension as applied but exact length unverified.

- [ ] **Step 5: Document the unchanged execution**

Add a short README section showing that the existing command is unchanged:

```powershell
python run_folder_analysis.py raw_sweep_260701_iris_tune_Torque13p5
```

Explain that campaign YAML is auto-discovered; users do not recreate TOML or directly maintain derived CSV.

- [ ] **Step 6: Run CLI integration tests**

Run: `python -m pytest tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py -q`

Expected: all tests pass.

---

### Task 4: Regression verification

**Files:**
- No new files.

**Interfaces:**
- Verifies all prior tasks together.

- [ ] **Step 1: Validate the real campaign against the current data tree**

Run a Python one-liner that calls `load_tuning_campaign("config/tuning_campaigns/iris_260701.yaml", data_root="data")` and prints its ID, baseline state, completed states, and pending states.

Expected: `iris_260701`, baseline `s000`, completed `s000/s001`, pending `s002`.

- [ ] **Step 2: Run the focused regression suite**

Run: `python -m pytest tests/test_tuning_campaign.py tests/test_tuning_campaign_workflow.py tests/test_workflow_manifest.py tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py -q`

Expected: all tests pass.

- [ ] **Step 3: Run formatting/static checks available in the repository**

Run: `python -m compileall deflector_tuning run_folder_analysis.py run_all_folder_analyses.py`

Expected: successful compilation with exit code zero.

- [ ] **Step 4: Inspect the final diff without modifying unrelated work**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; unrelated pre-existing modified and untracked files remain intact.
