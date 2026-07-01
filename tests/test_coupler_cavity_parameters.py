import math

import pandas as pd
import pytest

from deflector_tuning.analysis.coupler_cavity_parameters import (
    build_coupler_cavity_parameter_table,
    calculate_coupler_frequency_ghz,
    calculate_coupling_beta,
    calculate_external_quality_factor,
    calculate_matching_frequency_ghz,
    calculate_target_external_quality_factor,
    estimate_coupling_k_from_marker_frequencies,
    tan_half_phase,
)


def _reflection_phase_deg(frequency_ghz: float, *, coupler_frequency_ghz: float, external_q: float) -> float:
    tangent = (
        coupler_frequency_ghz
        * frequency_ghz
        / (external_q * (frequency_ghz**2 - coupler_frequency_ghz**2))
    )
    return math.degrees(2.0 * math.atan(tangent))


def _marker_rows(
    *,
    source_file: str,
    tune_position: float,
    reference_frequency_ghz: float = 10.0,
    operation_frequency_ghz: float = 12.0,
    reference_phase_deg: float = -12.0,
    operation_phase_deg: float = 13.0,
) -> list[dict[str, object]]:
    return [
        {
            "dataset_id": "dataset",
            "data_kind": "simulation",
            "data_layer": "sim",
            "source_file": source_file,
            "tune_position": tune_position,
            "s_name": "S11",
            "marker_role": "sim",
            "marker_name": "f_pi2",
            "target_freq_ghz": reference_frequency_ghz,
            "freq_ghz": reference_frequency_ghz,
            "s_phase_deg": reference_phase_deg,
            "sim_w_c": 18.0,
        },
        {
            "dataset_id": "dataset",
            "data_kind": "simulation",
            "data_layer": "sim",
            "source_file": source_file,
            "tune_position": tune_position,
            "s_name": "S11",
            "marker_role": "sim",
            "marker_name": "f_2pi3",
            "target_freq_ghz": operation_frequency_ghz,
            "freq_ghz": operation_frequency_ghz,
            "s_phase_deg": operation_phase_deg,
            "sim_w_c": 18.0,
        },
        {
            "dataset_id": "dataset",
            "data_kind": "simulation",
            "data_layer": "sim",
            "source_file": source_file,
            "tune_position": tune_position,
            "s_name": "S11",
            "marker_role": "sim",
            "marker_name": "f_mean",
            "target_freq_ghz": (reference_frequency_ghz + operation_frequency_ghz) / 2.0,
            "freq_ghz": (reference_frequency_ghz + operation_frequency_ghz) / 2.0,
            "s_phase_deg": 0.0,
            "sim_w_c": 18.0,
        },
    ]


def _geometry_marker_rows(
    *,
    source_file: str,
    sim_r_c: float,
    sim_w_c: float = 19.3224,
) -> list[dict[str, object]]:
    rows = _marker_rows(source_file=source_file, tune_position=float("nan"))
    for row in rows:
        row["tune_position"] = pd.NA
        row["scan_type"] = "geometry_sweep"
        row["sim_r_c"] = sim_r_c
        row["sim_w_c"] = sim_w_c
    return rows


def test_tan_half_phase_uses_degrees() -> None:
    assert tan_half_phase(60.0) == pytest.approx(1.0 / math.sqrt(3.0))


def test_coupler_frequency_and_external_quality_factor_recover_known_cavity() -> None:
    frequency_1_ghz = 10.0
    frequency_2_ghz = 12.0
    coupler_frequency_ghz = 11.0
    external_q = 50.0
    phase_1_deg = _reflection_phase_deg(
        frequency_1_ghz,
        coupler_frequency_ghz=coupler_frequency_ghz,
        external_q=external_q,
    )
    phase_2_deg = _reflection_phase_deg(
        frequency_2_ghz,
        coupler_frequency_ghz=coupler_frequency_ghz,
        external_q=external_q,
    )

    assert calculate_coupler_frequency_ghz(
        frequency_1_ghz,
        phase_1_deg,
        frequency_2_ghz,
        phase_2_deg,
    ) == pytest.approx(coupler_frequency_ghz)
    assert calculate_external_quality_factor(
        frequency_1_ghz,
        phase_1_deg,
        frequency_2_ghz,
        phase_2_deg,
    ) == pytest.approx(external_q)


def test_coupling_beta_uses_kyhl_half_phase_tangent_formula() -> None:
    frequency_1_ghz = 10.0
    frequency_2_ghz = 12.0
    phase_1_deg = -12.0
    phase_2_deg = 13.0
    coupling_k = 0.04
    pi_over_two_frequency_ghz = 10.0
    operation_mode_deg = 120.0
    t1 = tan_half_phase(phase_1_deg)
    t2 = tan_half_phase(phase_2_deg)
    expected = (
        1.0
        / ((coupling_k / 2.0) * pi_over_two_frequency_ghz * math.sin(math.radians(operation_mode_deg)))
        * (t1 * t2 * (frequency_1_ghz**2 - frequency_2_ghz**2))
        / (t2 * frequency_1_ghz - t1 * frequency_2_ghz)
    )

    assert calculate_coupling_beta(
        frequency_1_ghz,
        phase_1_deg,
        frequency_2_ghz,
        phase_2_deg,
        pi_over_two_frequency_ghz=pi_over_two_frequency_ghz,
        operation_mode_deg=operation_mode_deg,
        coupling_k=coupling_k,
    ) == pytest.approx(expected)


def test_matching_frequency_and_target_external_quality_factor_follow_zheng_targets() -> None:
    assert calculate_matching_frequency_ghz(10.0, 12.0) == pytest.approx(11.0)
    assert calculate_target_external_quality_factor(120.0, coupling_k=0.04) == pytest.approx(
        2.0 / (0.04 * math.sin(math.radians(120.0)))
    )


def test_estimate_coupling_k_from_marker_frequency_ratio_uses_operation_mode_cosine() -> None:
    assert estimate_coupling_k_from_marker_frequencies(
        pi_over_two_frequency_ghz=10.0,
        operation_frequency_ghz=10.1,
        operation_mode_deg=120.0,
    ) == pytest.approx(0.04)
    assert estimate_coupling_k_from_marker_frequencies(
        pi_over_two_frequency_ghz=10.0,
        operation_frequency_ghz=9.9,
        operation_mode_deg=120.0,
    ) == pytest.approx(0.04)


def test_build_coupler_cavity_parameter_table_pairs_pi2_and_operation_markers() -> None:
    frequency_1_ghz = 10.0
    frequency_2_ghz = 12.0
    phase_1_deg = _reflection_phase_deg(frequency_1_ghz, coupler_frequency_ghz=11.0, external_q=50.0)
    phase_2_deg = _reflection_phase_deg(frequency_2_ghz, coupler_frequency_ghz=11.0, external_q=50.0)
    marker_points = pd.DataFrame(
        [
            *_marker_rows(
                source_file="run_0p5.s1p",
                tune_position=0.5,
                reference_frequency_ghz=frequency_1_ghz,
                operation_frequency_ghz=frequency_2_ghz,
                reference_phase_deg=phase_1_deg,
                operation_phase_deg=phase_2_deg,
            ),
            *_marker_rows(
                source_file="run_1p5.s1p",
                tune_position=1.5,
                reference_frequency_ghz=frequency_1_ghz,
                operation_frequency_ghz=frequency_2_ghz,
                reference_phase_deg=phase_1_deg,
                operation_phase_deg=phase_2_deg,
            ),
        ]
    )
    markers = pd.DataFrame(
        [
            {"marker_name": "f_pi2", "freq_ghz": 10.0, "coupling_k": 0.04},
            {"marker_name": "f_2pi3", "freq_ghz": 12.0, "coupling_k": 0.04},
            {"marker_name": "f_mean", "freq_ghz": 11.0, "coupling_k": 0.04},
        ]
    )

    result = build_coupler_cavity_parameter_table(marker_points, markers)

    assert len(result) == 2
    assert result["tune_position"].tolist() == [0.5, 1.5]
    assert result["coupler_transition_pair"].unique().tolist() == ["0.5_to_1.5"]
    assert result["coupler_position_basis"].unique().tolist() == ["cell_center"]
    row = result[result["tune_position"] == 0.5].iloc[0]
    assert row["source_file"] == "run_0p5.s1p"
    assert row["reference_marker_name"] == "f_pi2"
    assert row["operation_marker_name"] == "f_2pi3"
    assert row["reference_frequency_ghz"] == pytest.approx(10.0)
    assert row["operation_frequency_ghz"] == pytest.approx(12.0)
    assert row["coupler_frequency_ghz"] == pytest.approx(11.0)
    assert row["external_quality_factor"] == pytest.approx(50.0)
    assert row["matching_frequency_ghz"] == pytest.approx(11.0)
    assert row["delta_frequency_mhz"] == pytest.approx(0.0)
    assert row["coupling_k"] == pytest.approx(0.04)
    assert row["coupling_k_source"] == "explicit_column"
    assert math.isfinite(row["coupling_beta"])
    assert math.isfinite(row["target_external_quality_factor"])
    assert row["is_valid"] is True
    assert row["invalid_reason"] == ""
    assert row["phase_input_convention"] == "raw_s11_reflection_phase_deg"


def test_build_coupler_cavity_parameter_table_derives_k_from_marker_frequency_ratio() -> None:
    marker_points = pd.DataFrame(
        [
            *_marker_rows(source_file="run_0p5.s1p", tune_position=0.5),
            *_marker_rows(source_file="run_1p5.s1p", tune_position=1.5),
        ]
    )
    markers = pd.DataFrame(
        [
            {"marker_name": "f_pi2", "freq_ghz": 10.0},
            {"marker_name": "f_2pi3", "freq_ghz": 12.0},
        ]
    )

    result = build_coupler_cavity_parameter_table(marker_points, markers)

    assert len(result) == 2
    row = result[result["tune_position"] == 0.5].iloc[0]
    assert math.isfinite(row["coupler_frequency_ghz"])
    assert math.isfinite(row["external_quality_factor"])
    assert row["coupling_k"] == pytest.approx(
        estimate_coupling_k_from_marker_frequencies(
            pi_over_two_frequency_ghz=10.0,
            operation_frequency_ghz=12.0,
            operation_mode_deg=120.0,
        )
    )
    assert row["coupling_k_source"] == "marker_frequency_ratio_abs"
    assert math.isfinite(row["coupling_beta"])
    assert math.isfinite(row["target_external_quality_factor"])
    assert row["coupling_beta_status"] == "ok"


def test_build_coupler_cavity_parameter_table_uses_only_complete_coupler_endpoint_pair() -> None:
    marker_points = pd.DataFrame(
        [
            *_marker_rows(source_file="run_0p5.s1p", tune_position=0.5),
            *_marker_rows(source_file="run_1p5.s1p", tune_position=1.5),
            *_marker_rows(source_file="run_9p5.s1p", tune_position=9.5),
            *_marker_rows(source_file="run_10p5.s1p", tune_position=10.5),
        ]
    )

    result = build_coupler_cavity_parameter_table(marker_points)

    assert result["tune_position"].tolist() == [0.5, 1.5]
    assert result["coupler_transition_pair"].tolist() == ["0.5_to_1.5", "0.5_to_1.5"]
    assert result["coupler_position_basis"].tolist() == ["cell_center", "cell_center"]
    assert result["coupler_pair_start_tune_position"].tolist() == [0.5, 0.5]
    assert result["coupler_pair_end_tune_position"].tolist() == [1.5, 1.5]


def test_build_coupler_cavity_parameter_table_supports_iris_center_pair() -> None:
    marker_points = pd.DataFrame(
        [
            *_marker_rows(source_file="run_0p0.s1p", tune_position=0.0),
            *_marker_rows(source_file="run_1p0.s1p", tune_position=1.0),
            *_marker_rows(source_file="run_2p0.s1p", tune_position=2.0),
        ]
    )

    result = build_coupler_cavity_parameter_table(marker_points)

    assert result["tune_position"].tolist() == [1.0, 2.0]
    assert result["coupler_transition_pair"].tolist() == ["1_to_2", "1_to_2"]
    assert result["coupler_position_basis"].tolist() == ["iris_center", "iris_center"]
    assert result["coupler_pair_start_tune_position"].tolist() == [1.0, 1.0]
    assert result["coupler_pair_end_tune_position"].tolist() == [2.0, 2.0]


def test_build_coupler_cavity_parameter_table_requires_complete_allowed_pair() -> None:
    marker_points = pd.DataFrame([*_marker_rows(source_file="run_0p5.s1p", tune_position=0.5)])

    result = build_coupler_cavity_parameter_table(marker_points)

    assert result.empty


def test_build_coupler_cavity_parameter_table_supports_geometry_radius_sweep_without_tune_positions() -> None:
    marker_points = pd.DataFrame(
        [
            *_geometry_marker_rows(source_file="rc_54.s1p", sim_r_c=54.0),
            *_geometry_marker_rows(source_file="rc_55.s1p", sim_r_c=55.0),
        ]
    )

    result = build_coupler_cavity_parameter_table(marker_points)

    assert len(result) == 2
    assert result["source_file"].tolist() == ["rc_54.s1p", "rc_55.s1p"]
    assert result["sim_r_c"].tolist() == [54.0, 55.0]
    assert result["sim_w_c"].tolist() == [19.3224, 19.3224]
    assert result["coupler_position_basis"].tolist() == ["geometry_sweep", "geometry_sweep"]
    assert result["coupler_transition_pair"].isna().all()
    assert result["coupling_beta_status"].tolist() == ["ok", "ok"]
