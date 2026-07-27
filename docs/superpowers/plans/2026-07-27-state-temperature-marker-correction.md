# State Temperature Marker Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make s003 and s004 marker sampling use their 23.4 C measurement metadata against the campaign's 30.0 C design reference, then regenerate their analysis products.

**Architecture:** Keep campaign metadata parsing and correction resolution in `deflector_tuning.tuning_campaign`.  The generic analysis runner receives an optional existing `TemperatureHumidityCorrection`; CLI and batch entry points resolve it before running analysis.  This preserves the current data-layer routing and avoids dataset-specific code branches.

**Tech Stack:** Python 3, PyYAML, pandas, NumPy, pytest.

## Global Constraints

- Preserve the current defaults for all YAML files and states without temperature metadata.
- Do not introduce dataset-name conditionals.
- Do not commit, push, or alter unrelated dirty-worktree changes.
- Regenerate only the existing s003 and s004 analysis outputs after tests pass.

---

### Task 1: Parse and resolve state marker-temperature metadata

**Files:**
- Modify: `deflector_tuning/tuning_campaign.py`
- Modify: `config/tuning_campaigns/iris_260701.yaml`
- Test: `tests/test_tuning_campaign.py`

**Interfaces:**
- Produces `MarkerCorrectionReference` and `TuningCampaign.marker_correction_for(state_id)`.
- `marker_correction_for` returns `TemperatureHumidityCorrection | None`.

- [ ] **Step 1: Write failing tests**

```python
def test_project_campaign_resolves_s003_marker_correction() -> None:
    campaign = load_tuning_campaign(PROJECT_CAMPAIGN_PATH)
    correction = campaign.marker_correction_for("s003")
    assert correction is not None
    assert correction.temp_op_C == pytest.approx(30.0)
    assert correction.temp_meas_C == pytest.approx(23.4)


def test_state_without_measurement_temperature_has_no_override(tmp_path: Path) -> None:
    campaign = load_tuning_campaign(_write_campaign(tmp_path))
    assert campaign.marker_correction_for("s001") is None
```

- [ ] **Step 2: Run the two tests and confirm they fail because the parser and resolver do not exist.**

Run: `python -m pytest tests/test_tuning_campaign.py -k marker_correction -q`

- [ ] **Step 3: Implement minimal metadata parsing and resolution.**

```python
@dataclass(frozen=True)
class MarkerCorrectionReference:
    design_temp_C: float = 20.0
    humidity_fraction: float = 0.65
    thermal_alpha_per_C: float = 1.68e-5
    eps_air_humid: float = 1.000712754221782


def marker_correction_for(self, state_id: str) -> TemperatureHumidityCorrection | None:
    measurement_temp_C = self.states[state_id].measurement_temp_C
    if measurement_temp_C is None:
        return None
    return TemperatureHumidityCorrection(...)
```

- [ ] **Step 4: Add campaign and state YAML metadata.**

```yaml
campaign:
  marker_correction:
    design_temp_C: 30.0
states:
  s003:
    temp_meas_C: 23.4
  s004:
    temp_meas_C: 23.4
```

- [ ] **Step 5: Re-run focused campaign tests.**

Run: `python -m pytest tests/test_tuning_campaign.py -q`

### Task 2: Forward resolved corrections into marker analysis

**Files:**
- Modify: `deflector_tuning/analysis/marker_pipeline.py`
- Modify: `deflector_tuning/runner.py`
- Test: `tests/test_marker_analysis_pipeline.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- `build_marker_analysis(..., marker_correction: TemperatureHumidityCorrection | None = None)`.
- `run_folder_analysis(..., marker_correction: TemperatureHumidityCorrection | None = None)`.

- [ ] **Step 1: Write failing forwarding tests.**

```python
def test_build_marker_analysis_forwards_explicit_marker_correction(monkeypatch, tmp_path):
    correction = TemperatureHumidityCorrection(temp_op_C=30.0, temp_meas_C=23.4)
    ...
    build_marker_analysis(..., marker_correction=correction)
    assert captured["correction"] is correction
```

- [ ] **Step 2: Run the forwarding tests and confirm they fail because the keyword is unsupported.**

Run: `python -m pytest tests/test_marker_analysis_pipeline.py -k correction -q`

- [ ] **Step 3: Implement optional forwarding without changing the default path.**

```python
markers = extract_marker_frequencies(
    dispersion_path,
    marker_role=marker_role,
    correction=marker_correction,
)
```

- [ ] **Step 4: Re-run the focused marker and runner tests.**

Run: `python -m pytest tests/test_marker_analysis_pipeline.py tests/test_runner.py -q`

### Task 3: Resolve corrections in CLI and batch workflows, and record provenance

**Files:**
- Modify: `deflector_tuning/workflows/tuning_campaign.py`
- Modify: `deflector_tuning/workflows/manifest.py`
- Modify: `run_folder_analysis.py`
- Modify: `run_all_folder_analyses.py`
- Test: `tests/test_tuning_campaign_workflow.py`
- Test: `tests/test_run_folder_analysis_cli.py`
- Test: `tests/test_run_all_folder_analyses_cli.py`

**Interfaces:**
- `marker_correction_for_tuning_dataset(dataset_id, campaign_dir, data_root)` returns a resolved correction and optional campaign match.
- Manifest `tuning_campaign.marker_correction` contains the state-resolved numerical provenance or `null`.

- [ ] **Step 1: Write failing workflow tests.**

```python
def test_tuning_dataset_resolves_state_marker_correction(tmp_path):
    match = find_tuning_campaign(...)
    correction = marker_correction_for_tuning_dataset(...)
    assert correction.temp_op_C == pytest.approx(30.0)
    assert correction.temp_meas_C == pytest.approx(23.4)
```

- [ ] **Step 2: Run the workflow tests and confirm they fail because no resolver exists.**

Run: `python -m pytest tests/test_tuning_campaign_workflow.py -k correction -q`

- [ ] **Step 3: Implement pre-analysis resolution and manifest provenance.**

```python
correction = marker_correction_for_tuning_dataset(dataset_id, data_root=data_root)
result = run_folder_analysis(..., marker_correction=correction)
register_matching_tuning_campaign(..., marker_correction=correction)
```

- [ ] **Step 4: Re-run focused workflow and CLI tests.**

Run: `python -m pytest tests/test_tuning_campaign_workflow.py tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py -q`

### Task 4: Verify and regenerate s003 and s004

**Files:**
- Regenerate: `fig/analyses/raw_sweep_260721_tune_s003_broken/`
- Regenerate: `fig/analyses/raw_sweep_260722_tune_s004/`

- [ ] **Step 1: Run all affected tests.**

Run: `python -m pytest tests/test_tuning_campaign.py tests/test_marker_analysis_pipeline.py tests/test_tuning_campaign_workflow.py tests/test_run_folder_analysis_cli.py tests/test_run_all_folder_analyses_cli.py -q`

- [ ] **Step 2: Regenerate s003 in its established output directory.**

Run: `python run_folder_analysis.py raw/raw_sweep_260721_tune_s003_broken --output-dir fig/analyses/raw_sweep_260721_tune_s003_broken`

- [ ] **Step 3: Regenerate s004 in its established output directory.**

Run: `python run_folder_analysis.py raw/raw_sweep_260722_tune_s004 --output-dir fig/analyses/raw_sweep_260722_tune_s004`

- [ ] **Step 4: Verify regenerated marker tables and manifests.**

```python
for dataset_id in ("raw_sweep_260721_tune_s003_broken", "raw_sweep_260722_tune_s004"):
    markers = pd.read_csv(Path("fig/analyses") / dataset_id / "tables" / "markers.csv")
    assert set(markers["temp_op_C"]) == {30.0}
    assert set(markers["temp_meas_C"]) == {23.4}
```

- [ ] **Step 5: Inspect the final diff without staging or committing.**

Run: `git diff --check`
