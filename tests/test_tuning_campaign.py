from __future__ import annotations

from pathlib import Path

import pytest

from deflector_tuning.tuning_campaign import (
    find_tuning_campaign,
    is_tuning_dataset_id,
    load_tuning_campaign,
    tuning_torque_nm_from_dataset_id,
)


COMPACT_CAMPAIGN = """
schema: 1
campaign:
  id: iris_260701
  sim: sim_grid_260701_1DRcFine
  rc_design_mm: 56.59
  phase:
    port_ext: applied
    ext_mm: null
    verified: false
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
issues:
  i001:
    after: s001
    verify: s002
    tag: input.lr_tuner_disconnect
    check: [output]
    note: tuner bolt disconnected from internal radius-adjust screw
""".strip()


def _write_campaign(tmp_path: Path, text: str = COMPACT_CAMPAIGN, name: str = "iris.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _make_data_tree(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    (data_root / "sim" / "sim_grid_260701_1DRcFine").mkdir(parents=True)
    (data_root / "raw" / "raw_sweep_260701_iris_portE").mkdir(parents=True)
    (data_root / "raw" / "raw_sweep_260701_iris_tune_Torque13p5").mkdir(parents=True)
    return data_root


@pytest.mark.parametrize(
    ("dataset_id", "state_id", "measurement_kind", "comparison_enabled"),
    [
        ("raw_sweep_260721_tune_s002_Torque13p5", "s002", "state", True),
        ("raw_sweep_260721_tune_s003_broken", "s003", "state", True),
        (
            "raw_sweep_260721_tune_s003_plungersensitivity",
            "s003",
            "aux",
            False,
        ),
    ],
)
def test_project_iris_campaign_registers_completed_tuning_states(
    dataset_id: str,
    state_id: str,
    measurement_kind: str,
    comparison_enabled: bool,
) -> None:
    campaign_dir = Path(__file__).resolve().parents[1] / "config" / "tuning_campaigns"

    match = find_tuning_campaign(dataset_id, campaign_dir)

    assert match is not None
    assert match.campaign.campaign_id == "iris_260701"
    assert match.state_id == state_id
    assert match.measurement_kind == measurement_kind
    assert match.comparison_enabled is comparison_enabled


def test_load_tuning_campaign_expands_compact_issue_defaults(tmp_path: Path) -> None:
    campaign = load_tuning_campaign(_write_campaign(tmp_path))

    assert campaign.campaign_id == "iris_260701"
    assert campaign.simulation_dataset == "sim_grid_260701_1DRcFine"
    assert campaign.design_r_c_mm == pytest.approx(56.59)
    assert campaign.baseline_state_id == "s000"
    assert tuple(campaign.states) == ("s000", "s001", "s002")
    assert campaign.states["s001"].torque_nm == pytest.approx(13.5)
    assert campaign.states["s001"].quality_flag == "provisional"
    assert campaign.states["s002"].dataset is None
    assert campaign.states["s002"].measurement_status == "pending"
    assert campaign.phase.port_extension == "applied"
    assert campaign.phase.extension_mm is None
    assert campaign.phase.verified is False

    issue = campaign.issues["i001"]
    assert issue.status == "open"
    assert issue.severity == "warn"
    assert issue.action == "flag"
    assert issue.check == ("output",)
    assert issue.after_state_id == "s001"
    assert issue.verification_state_id == "s002"


def test_load_tuning_campaign_reads_role_keyed_rc_sweeps(tmp_path: Path) -> None:
    text = COMPACT_CAMPAIGN.replace(
        "  sim: sim_grid_260701_1DRcFine",
        """  sims:
    iris:
      data: sim_grid_260701_1DRcFine
      exp_pos: [1, 2]
      axis: r_c
    cell:
      data: sim_grid_260720_1DRcFine_Cell
      exp_pos: [0.5, 1.5]
      axis: r_c""",
    )

    campaign = load_tuning_campaign(_write_campaign(tmp_path, text))

    assert tuple(campaign.simulation_references) == ("iris", "cell")
    assert campaign.simulation_references["iris"].dataset == "sim_grid_260701_1DRcFine"
    assert campaign.simulation_references["iris"].experiment_positions == (1.0, 2.0)
    assert campaign.simulation_references["cell"].experiment_positions == (0.5, 1.5)
    assert campaign.simulation_references["cell"].axis == "r_c"


def test_load_tuning_campaign_rejects_non_rc_automatic_sweep(tmp_path: Path) -> None:
    text = COMPACT_CAMPAIGN.replace(
        "  sim: sim_grid_260701_1DRcFine",
        """  sims:
    iris:
      data: sim_grid_260701_1DRcFine
      exp_pos: [1, 2]
      axis: tuner_insertion_depth""",
    )

    with pytest.raises(ValueError, match="axis.*r_c"):
        load_tuning_campaign(_write_campaign(tmp_path, text))


def test_load_tuning_campaign_allows_pending_state_without_dataset(tmp_path: Path) -> None:
    campaign = load_tuning_campaign(_write_campaign(tmp_path))

    assert campaign.states["s002"].measurement_status == "pending"
    assert campaign.states["s002"].dataset is None


def test_load_tuning_campaign_requires_dataset_for_completed_state(tmp_path: Path) -> None:
    text = COMPACT_CAMPAIGN.replace(
        "  s002:\n    prev: s001\n    meas: pending",
        "  s002:\n    prev: s001\n    meas: done",
    )

    with pytest.raises(ValueError, match="s002.*data"):
        load_tuning_campaign(_write_campaign(tmp_path, text))


def test_load_tuning_campaign_requires_one_baseline(tmp_path: Path) -> None:
    text = COMPACT_CAMPAIGN.replace("    role: baseline\n", "", 1)

    with pytest.raises(ValueError, match="exactly one baseline"):
        load_tuning_campaign(_write_campaign(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("    prev: s001", "    prev: missing", "s002.*prev.*missing"),
        ("    after: s001", "    after: missing", "i001.*after.*missing"),
        ("    verify: s002", "    verify: missing", "i001.*verify.*missing"),
    ],
)
def test_load_tuning_campaign_rejects_missing_state_references(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_tuning_campaign(_write_campaign(tmp_path, COMPACT_CAMPAIGN.replace(old, new)))


def test_load_tuning_campaign_rejects_previous_state_cycle(tmp_path: Path) -> None:
    text = COMPACT_CAMPAIGN.replace(
        "  s000:\n    role: baseline",
        "  s000:\n    role: baseline\n    prev: s002",
    )

    with pytest.raises(ValueError, match="cycle"):
        load_tuning_campaign(_write_campaign(tmp_path, text))


def test_load_tuning_campaign_rejects_duplicate_completed_dataset(tmp_path: Path) -> None:
    text = COMPACT_CAMPAIGN.replace(
        "data: raw_sweep_260701_iris_tune_Torque13p5",
        "data: raw_sweep_260701_iris_portE",
    )

    with pytest.raises(ValueError, match="duplicate completed-state dataset"):
        load_tuning_campaign(_write_campaign(tmp_path, text))


def test_load_tuning_campaign_validates_completed_dataset_locations(tmp_path: Path) -> None:
    data_root = _make_data_tree(tmp_path)
    campaign = load_tuning_campaign(_write_campaign(tmp_path), data_root=data_root)
    assert campaign.baseline_state_id == "s000"

    missing_data_root = tmp_path / "missing_data"
    missing_data_root.mkdir()
    with pytest.raises(FileNotFoundError, match="data.sim.*sim_grid_260701_1DRcFine"):
        load_tuning_campaign(_write_campaign(tmp_path, name="missing.yaml"), data_root=missing_data_root)


def test_find_tuning_campaign_matches_state_or_simulation_dataset(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    _write_campaign(campaign_dir)

    by_state = find_tuning_campaign(
        "raw_sweep_260701_iris_tune_Torque13p5", campaign_dir
    )
    by_sim = find_tuning_campaign("sim_grid_260701_1DRcFine", campaign_dir)

    assert by_state is not None
    assert by_state.state_id == "s001"
    assert by_state.measurement_kind == "state"
    assert by_state.comparison_enabled is True
    assert by_sim is not None
    assert by_sim.state_id is None
    assert by_sim.measurement_kind == "simulation"
    assert by_sim.comparison_enabled is False
    assert find_tuning_campaign("raw_sweep_260701_unrelated", campaign_dir) is None


def test_find_tuning_campaign_rejects_ambiguous_dataset_match(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    _write_campaign(campaign_dir, name="first.yaml")
    _write_campaign(
        campaign_dir,
        COMPACT_CAMPAIGN.replace("id: iris_260701", "id: iris_other"),
        name="second.yml",
    )

    with pytest.raises(ValueError, match="multiple tuning campaigns"):
        find_tuning_campaign("raw_sweep_260701_iris_portE", campaign_dir)


def test_find_tuning_campaign_does_not_validate_unmatched_campaign_data(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    _write_campaign(campaign_dir)
    empty_data_root = tmp_path / "empty_data"
    empty_data_root.mkdir()

    assert (
        find_tuning_campaign(
            "raw_sweep_260701_unrelated",
            campaign_dir,
            data_root=empty_data_root,
        )
        is None
    )


def test_find_tuning_campaign_ignores_invalid_unrelated_campaign(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    (campaign_dir / "broken.yaml").write_text(
        "schema: invalid\ncampaign:\n  id: broken\n  sim: sim_other\nstates: {}\n",
        encoding="utf-8",
    )

    assert find_tuning_campaign("raw_sweep_260701_unrelated", campaign_dir) is None


@pytest.mark.parametrize(
    ("dataset_id", "expected"),
    [
        ("raw_sweep_260701_iris_tune_Torque13p5", True),
        ("raw_sweep_260701_iris_TUNE_torque17", True),
        ("raw_sweep_260701_iris_portE", False),
        ("raw_sweep_260701_iris_tuneTorque13p5", False),
        ("raw_sweep_260701_iris_tuner_check", False),
    ],
)
def test_is_tuning_dataset_id_requires_tune_token(
    dataset_id: str, expected: bool
) -> None:
    assert is_tuning_dataset_id(dataset_id) is expected


def test_tuning_torque_nm_from_dataset_id_parses_p_decimal_case_insensitively() -> None:
    assert tuning_torque_nm_from_dataset_id(
        "raw_sweep_260701_iris_tune_Torque13p5"
    ) == pytest.approx(13.5)
    assert tuning_torque_nm_from_dataset_id(
        "raw_sweep_260701_iris_tune_torque17"
    ) == pytest.approx(17.0)
    assert tuning_torque_nm_from_dataset_id(
        "raw_sweep_260701_iris_portE"
    ) is None


def test_find_tuning_campaign_infers_next_state_for_unique_active_campaign(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns"
    campaign_dir.mkdir()
    text = COMPACT_CAMPAIGN.replace(
        "  id: iris_260701",
        "  id: iris_260701\n  auto_states: true",
    ).replace(
        "  s002:\n    prev: s001\n    meas: pending\n    change: tuner_bolt_added_only",
        """  s002:
    prev: s001
    data: raw_sweep_260721_tune_s002_Torque13p5
    meas: done
  s003:
    prev: s002
    data: raw_sweep_260721_tune_s003_broken
    meas: done""",
    )
    _write_campaign(campaign_dir, text)

    match = find_tuning_campaign("raw_sweep_260722_tune_s004", campaign_dir)

    assert match is not None
    assert match.state_id == "s004"
    assert match.measurement_kind == "state"
    assert match.match_mode == "auto"
    state = match.campaign.states["s004"]
    assert state.previous_state_id == "s003"
    assert state.dataset == "raw_sweep_260722_tune_s004"
    assert state.quality_flag == "auto_pending"


def test_find_tuning_campaign_keeps_explicit_auxiliary_mapping_before_auto_matching() -> None:
    campaign_dir = Path(__file__).resolve().parents[1] / "config" / "tuning_campaigns"

    match = find_tuning_campaign(
        "raw_sweep_260721_tune_s003_plungersensitivity",
        campaign_dir,
    )

    assert match is not None
    assert match.state_id == "s003"
    assert match.measurement_kind == "aux"
    assert match.match_mode == "explicit"
