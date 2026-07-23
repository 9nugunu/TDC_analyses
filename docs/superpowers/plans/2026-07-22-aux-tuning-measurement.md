# Auxiliary Tuning Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run supplementary tuning measurements without inventing a mechanical tuning state or an `r_c` comparison.

**Architecture:** Parse a compact campaign-level `aux` mapping into an auxiliary-measurement model.  A campaign match carries the parent state and a generic comparison-enabled flag; the CLI uses that flag to decide whether to run the simulation comparison.

**Tech Stack:** Python 3.11, PyYAML, pytest.

## Global Constraints

- Keep `states` as the ordered mechanical-state record.
- Do not add dataset-name-specific branches.
- `cmp: false` must skip only automatic tuning comparison, not the normal folder analysis.

---

### Task 1: Parse and match auxiliary measurements

**Files:**
- Modify: `deflector_tuning/tuning_campaign.py`
- Modify: `tests/test_tuning_campaign.py`

**Interfaces:**
- Consumes: campaign YAML `aux.<dataset>.state` and `aux.<dataset>.cmp`.
- Produces: `TuningCampaignMatch.state_id` and `TuningCampaignMatch.comparison_enabled`.

- [ ] **Step 1: Write failing tests**

```python
match = find_tuning_campaign("raw_sweep_260721_tune_s003_plungersensitivity", campaign_dir)
assert match.state_id == "s003"
assert match.comparison_enabled is False
```

- [ ] **Step 2: Run the focused test and verify it fails because `aux` is not recognized.**

- [ ] **Step 3: Add the compact auxiliary-measurement dataclass, YAML parsing, validation, and matching.**

- [ ] **Step 4: Run the focused test and verify it passes.**

### Task 2: Register the measurement and skip only its comparison

**Files:**
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `deflector_tuning/workflows/tuning_campaign.py`
- Modify: `run_folder_analysis.py`
- Modify: `tests/test_tuning_campaign_workflow.py`
- Modify: `tests/test_run_folder_analysis_cli.py`

**Interfaces:**
- Consumes: `TuningCampaignMatch.comparison_enabled`.
- Produces: manifest `tuning_campaign.measurement_kind` and a no-comparison CLI path for `cmp: false`.

- [ ] **Step 1: Write failing tests**

```python
assert manifest["tuning_campaign"]["measurement_kind"] == "aux"
assert comparisons == []
```

- [ ] **Step 2: Run the focused tests and verify they fail.**

- [ ] **Step 3: Record the measurement kind and guard `run_tuning_cmp` with `comparison_enabled`.**

- [ ] **Step 4: Run focused workflow and CLI tests and verify they pass.**

### Task 3: Register the current sensitivity dataset and smoke-test it

**Files:**
- Modify: `config/tuning_campaigns/iris_260701.yaml`
- Modify: `tests/test_tuning_campaign.py`

- [ ] **Step 1: Add the `s003` sensitivity dataset under `aux` with `cmp: false`.**

- [ ] **Step 2: Run campaign tests and a `--tables-only` CLI smoke test.**

- [ ] **Step 3: Confirm the manifest records `s003` and `measurement_kind: aux`, with no `tuning_cmp` output.**
