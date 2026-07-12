from pathlib import Path
import pickle

from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS, load_project_defaults


def test_project_defaults_collect_scientific_analysis_assumptions() -> None:
    defaults = DEFAULT_PROJECT_DEFAULTS

    assert defaults.default_dispersion_subpath == Path("sim") / "sim_dispersion_260505_single_cell_step1"
    assert defaults.design_point_by_axis == {"sim_r_c": 56.59, "sim_w_c": 19.3224}
    assert defaults.ideal_phase_guide_angles_deg == (0.0, 180.0, 60.0, -60.0)


def test_load_project_defaults_reads_user_editable_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "project_defaults.toml"
    config_path.write_text(
        """
[analysis]
default_dispersion_subpath = "sim/custom_dispersion"

[visualization]
ideal_phase_guide_angles_deg = [0.0, 120.0]

[visualization.design_point_by_axis]
sim_r_c = 57.0
sim_w_c = 20.0
""".strip(),
        encoding="utf-8",
    )

    defaults = load_project_defaults(config_path)

    assert defaults.default_dispersion_subpath == Path("sim") / "custom_dispersion"
    assert defaults.design_point_by_axis == {"sim_r_c": 57.0, "sim_w_c": 20.0}
    assert defaults.ideal_phase_guide_angles_deg == (0.0, 120.0)


def test_project_defaults_can_be_sent_to_batch_worker_processes() -> None:
    restored = pickle.loads(pickle.dumps(DEFAULT_PROJECT_DEFAULTS))

    assert restored == DEFAULT_PROJECT_DEFAULTS
