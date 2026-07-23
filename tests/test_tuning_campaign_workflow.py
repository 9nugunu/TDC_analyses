from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.tuning_campaign import load_tuning_campaign
from deflector_tuning.workflows.manifest import (
    register_tuning_campaign,
    register_tuning_cmp,
)
from deflector_tuning.workflows.tuning_campaign import (
    campaign_issue_table,
    campaign_state_table,
    export_campaign_tables,
    register_matching_tuning_campaign,
)


CAMPAIGN_YAML = """
schema: 1
campaign:
  id: iris_260701
  sim: sim_grid_260701_1DRcFine
  rc_design_mm: 56.59
states:
  s000:
    role: baseline
    data: raw_sweep_260701_iris_portE
    torque_nm: 0
    meas: done
  s001:
    prev: s000
    data: raw_sweep_260701_iris_tune_Torque13p5
    torque_nm: 13.5
    meas: done
    flag: provisional
  s002:
    prev: s001
    meas: pending
    change: tuner_bolt_added_only
aux:
  raw_sweep_260721_tune_s002_plungersensitivity:
    state: s002
    cmp: false
issues:
  i001:
    after: s001
    verify: s002
    tag: input.lr_tuner_disconnect
    check: [output]
    note: output coupler impact is unknown
""".strip()


def _campaign(tmp_path: Path):
    path = tmp_path / "iris.yaml"
    path.write_text(CAMPAIGN_YAML, encoding="utf-8")
    return load_tuning_campaign(path)


def test_campaign_state_table_keeps_pending_state_and_compact_fields(tmp_path: Path) -> None:
    table = campaign_state_table(_campaign(tmp_path))

    assert table["state_id"].tolist() == ["s000", "s001", "s002"]
    assert table["seq"].tolist() == [0, 1, 2]
    pending = table.loc[table["state_id"] == "s002"].iloc[0]
    assert pending["meas"] == "pending"
    assert pd.isna(pending["data"])
    assert pending["change"] == "tuner_bolt_added_only"


def test_campaign_issue_table_expands_defaults_without_inventing_impact(tmp_path: Path) -> None:
    table = campaign_issue_table(_campaign(tmp_path))

    assert table.to_dict("records") == [
        {
            "issue_id": "i001",
            "after": "s001",
            "verify": "s002",
            "tag": "input.lr_tuner_disconnect",
            "check": "output",
            "stat": "open",
            "sev": "warn",
            "action": "flag",
            "note": "output coupler impact is unknown",
        }
    ]
    assert "affected" not in table.columns


def test_export_campaign_tables_writes_derived_csv_without_rewriting_yaml(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    original_yaml = campaign.source_path.read_bytes()

    outputs = export_campaign_tables(campaign, tmp_path / "data" / "prepro" / "iris_260701")

    assert set(outputs) == {"states", "issues"}
    assert outputs["states"].name == "states.csv"
    assert outputs["issues"].name == "issues.csv"
    assert pd.read_csv(outputs["states"])["state_id"].tolist() == ["s000", "s001", "s002"]
    assert campaign.source_path.read_bytes() == original_yaml


def test_register_tuning_campaign_preserves_manifest_and_links_issues(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"analysis_modes": ["raw"], "outputs": {"tables": {"markers": "x.csv"}}}),
        encoding="utf-8",
    )

    register_tuning_campaign(
        manifest_path,
        config_path=Path("config/tuning_campaigns/iris_260701.yaml"),
        campaign=campaign,
        state_id="s001",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis_modes"] == ["raw"]
    assert manifest["outputs"]["tables"] == {"markers": "x.csv"}
    assert manifest["tuning_campaign"] == {
        "campaign_id": "iris_260701",
        "config_path": "config/tuning_campaigns/iris_260701.yaml",
        "state_id": "s001",
        "state_status": "done",
        "measurement_kind": "state",
        "issue_ids": ["i001"],
    }


def test_register_tuning_cmp_preserves_standard_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"outputs": {"tables": {"marker_pts": "marker_pts.csv"}, "figures": {}}}),
        encoding="utf-8",
    )

    register_tuning_cmp(
        manifest_path,
        tables={"phase_cmp": tmp_path / "phase_cmp.csv"},
        figures={"phase_rc_map": tmp_path / "phase_rc_map.png"},
        metadata={"families": ["iris"], "baseline_state_id": "s000"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["outputs"]["tables"]["marker_pts"] == "marker_pts.csv"
    assert manifest["outputs"]["tuning_cmp"]["tables"] == {
        "phase_cmp": str(tmp_path / "phase_cmp.csv")
    }
    assert manifest["outputs"]["tuning_cmp"]["figures"] == {
        "phase_rc_map": str(tmp_path / "phase_rc_map.png")
    }
    assert manifest["tuning_cmp"]["families"] == ["iris"]


def test_register_matching_tuning_campaign_is_a_noop_for_unmatched_dataset(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    (campaign_dir / "iris.yaml").write_text(CAMPAIGN_YAML, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    original = {"analysis_modes": ["raw"]}
    manifest_path.write_text(json.dumps(original), encoding="utf-8")

    match = register_matching_tuning_campaign(
        "raw_sweep_260701_unrelated",
        manifest_path=manifest_path,
        campaign_dir=campaign_dir,
    )

    assert match is None
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_register_matching_tuning_campaign_adds_matching_state(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    (campaign_dir / "iris.yaml").write_text(CAMPAIGN_YAML, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"analysis_modes": ["raw"]}), encoding="utf-8")

    match = register_matching_tuning_campaign(
        "raw_sweep_260701_iris_tune_Torque13p5",
        manifest_path=manifest_path,
        campaign_dir=campaign_dir,
    )

    assert match is not None
    assert match.state_id == "s001"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tuning_campaign"]["state_id"] == "s001"


def test_register_matching_tuning_campaign_links_auxiliary_measurement(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    (campaign_dir / "iris.yaml").write_text(CAMPAIGN_YAML, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"analysis_modes": ["raw"]}), encoding="utf-8")

    match = register_matching_tuning_campaign(
        "raw_sweep_260721_tune_s002_plungersensitivity",
        manifest_path=manifest_path,
        campaign_dir=campaign_dir,
    )

    assert match is not None
    assert match.state_id == "s002"
    assert match.measurement_kind == "aux"
    assert match.comparison_enabled is False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tuning_campaign"]["state_id"] == "s002"
    assert manifest["tuning_campaign"]["measurement_kind"] == "aux"


def test_register_matching_tuning_campaign_skips_registered_non_tuning_dataset(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    (campaign_dir / "iris.yaml").write_text(CAMPAIGN_YAML, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    original = {"analysis_modes": ["raw"]}
    manifest_path.write_text(json.dumps(original), encoding="utf-8")

    match = register_matching_tuning_campaign(
        "raw_sweep_260701_iris_portE",
        manifest_path=manifest_path,
        campaign_dir=campaign_dir,
    )

    assert match is None
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_register_matching_tuning_campaign_rejects_unregistered_tuning_dataset(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"analysis_modes": ["raw"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="tuning dataset.*not registered"):
        register_matching_tuning_campaign(
            "raw_sweep_260721_iris_tune_Torque17",
            manifest_path=manifest_path,
            campaign_dir=campaign_dir,
        )


def test_register_matching_tuning_campaign_rejects_folder_yaml_torque_mismatch(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    mismatch = CAMPAIGN_YAML.replace("torque_nm: 13.5", "torque_nm: 14.0")
    (campaign_dir / "iris.yaml").write_text(mismatch, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"analysis_modes": ["raw"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="folder torque 13.5.*YAML torque 14.0"):
        register_matching_tuning_campaign(
            "raw_sweep_260701_iris_tune_Torque13p5",
            manifest_path=manifest_path,
            campaign_dir=campaign_dir,
        )
