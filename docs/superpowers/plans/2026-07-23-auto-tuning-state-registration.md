# Automatic Tuning-State Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically record and compare the next `*_tune_sNNN` state in the unique active tuning campaign.

**Architecture:** Campaign YAML declares one active automatic campaign.  The campaign resolver first honors exact state/auxiliary entries, then infers a consecutive primary state or `_aux_plunger` measurement and records a compact YAML entry only after the ordinary analysis succeeds.  The manifest records whether the match was automatic.

**Tech Stack:** Python 3.12, PyYAML, pytest.

## Global Constraints

- `data/raw` remains the raw-data source; no per-dataset metadata file is created.
- Campaign YAML is the central state record; CSV exports remain derived views.
- Exact YAML mappings take precedence over inference.
- An inferred primary state may only be the immediate successor of the current highest numeric state.

---

### Task 1: Parse and resolve automatic campaign matches

**Files:**
- Modify: `deflector_tuning/tuning_campaign.py`
- Modify: `tests/test_tuning_campaign.py`

**Interfaces:**
- Produces `find_tuning_campaign(dataset_id, campaign_dir, ...)` matches for explicit and inferred datasets.
- Adds `auto_states_enabled` and `match_mode` to campaign/match metadata.

- [ ] **Step 1: Write failing resolver tests**

```python
def test_find_tuning_campaign_infers_next_state_for_unique_active_campaign(tmp_path):
    campaign_dir = _write_campaign(tmp_path, auto_states=True, last_state="s003")
    match = find_tuning_campaign("raw_sweep_260722_tune_s004", campaign_dir)
    assert match.state_id == "s004"
    assert match.measurement_kind == "state"
    assert match.match_mode == "auto"
    assert match.campaign.states["s004"].previous_state_id == "s003"

def test_find_tuning_campaign_keeps_explicit_auxiliary_mapping(tmp_path):
    campaign_dir = _write_campaign(tmp_path, explicit_s003_aux=True, auto_states=True)
    match = find_tuning_campaign("raw_sweep_260721_tune_s003_plungersensitivity", campaign_dir)
    assert match.state_id == "s003"
    assert match.measurement_kind == "aux"
    assert match.match_mode == "explicit"
```

- [ ] **Step 2: Run the resolver tests and verify RED**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_tuning_campaign.py -k auto`

Expected: FAIL because `auto_states` and `match_mode` do not exist.

- [ ] **Step 3: Implement the minimal resolver**

Add an optional `campaign.auto_states` boolean, parse `tune_sNNN`, retain the current exact-match pass, and only then select one auto-enabled campaign.  Create an immutable derived `TuningState` with `prev` set to `s(N-1)`, `meas: done`, `flag: auto_pending`, and a parsed folder torque when present.  Reject state-ID collisions, skipped state numbers, and multiple auto campaigns.

- [ ] **Step 4: Run resolver tests and verify GREEN**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_tuning_campaign.py`

Expected: PASS.

### Task 2: Persist a newly inferred primary state and manifest provenance

**Files:**
- Modify: `deflector_tuning/workflows/tuning_campaign.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `tests/test_tuning_campaign_workflow.py`

**Interfaces:**
- `register_matching_tuning_campaign(...)` writes a new automatic primary-state entry to the matched YAML only after ordinary analysis has completed.
- `register_tuning_campaign(..., match_mode=...)` adds `match_mode` to the manifest campaign section.

- [ ] **Step 1: Write failing workflow tests**

```python
def test_register_matching_tuning_campaign_persists_inferred_state(tmp_path):
    match = register_matching_tuning_campaign(
        "raw_sweep_260722_tune_s004", manifest_path=manifest_path, campaign_dir=campaign_dir
    )
    recorded = yaml.safe_load((campaign_dir / "iris.yaml").read_text())
    assert recorded["states"]["s004"] == {
        "prev": "s003", "data": "raw_sweep_260722_tune_s004",
        "meas": "done", "flag": "auto_pending",
    }
    assert manifest["tuning_campaign"]["match_mode"] == "auto"

def test_register_matching_tuning_campaign_is_idempotent_for_auto_state(tmp_path):
    register_matching_tuning_campaign("raw_sweep_260722_tune_s004", ...)
    register_matching_tuning_campaign("raw_sweep_260722_tune_s004", ...)
    assert list(recorded["states"]) == ["s000", "s001", "s002", "s003", "s004"]
```

- [ ] **Step 2: Run workflow tests and verify RED**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_tuning_campaign_workflow.py -k "inferred or idempotent"`

Expected: FAIL because the YAML is unchanged and no manifest provenance exists.

- [ ] **Step 3: Implement atomic YAML registration and manifest provenance**

Use `yaml.safe_load` and `yaml.safe_dump(sort_keys=False, allow_unicode=True)` through a temporary sibling file and replacement.  Persist only an inferred primary state.  Re-analysis of the same recorded state becomes explicit and does not write another entry.  Add `match_mode` to the manifest without modifying existing output entries.

- [ ] **Step 4: Run workflow tests and verify GREEN**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_tuning_campaign_workflow.py`

Expected: PASS.

### Task 3: Enable the active campaign and verify the real S004 path

**Files:**
- Modify: `config/tuning_campaigns/iris_260701.yaml`
- Modify: `tests/test_run_folder_analysis_cli.py`

**Interfaces:**
- `iris_260701.yaml` is the one active automatic campaign.
- `run_folder_analysis.py data/raw/raw_sweep_260722_tune_s004` completes and writes comparison outputs.

- [ ] **Step 1: Write the failing CLI-level campaign test**

```python
def test_main_allows_next_automatic_tuning_state(tmp_path, monkeypatch):
    result = main([str(next_tuning_folder), "--data-root", str(data_root)])
    assert result == 0
    assert registered_campaign_yaml["states"]["s004"]["data"] == next_tuning_folder.name
```

- [ ] **Step 2: Run the CLI test and verify RED**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_run_folder_analysis_cli.py -k automatic`

Expected: FAIL with the current unregistered tuning-dataset error.

- [ ] **Step 3: Declare `auto_states: true` in the campaign YAML**

Add `auto_states: true` beside `campaign.id`; do not add S004 manually.

- [ ] **Step 4: Run focused and full verification**

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q tests/test_tuning_campaign.py tests/test_tuning_campaign_workflow.py tests/test_run_folder_analysis_cli.py`

Expected: PASS.

Run: `C:\Users\9nugu\miniconda3\envs\sys_env1\python.exe -m pytest -q`

Expected: PASS.
