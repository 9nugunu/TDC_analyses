from pathlib import Path
import json

from deflector_tuning.workflows.manifest import cached_manifest_figures, write_manifest


def test_cached_manifest_figures_resolves_project_relative_paths(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "fig" / "analyses" / "case"
    figure = output_dir / "figures" / "s11" / "overview.png"
    figure.parent.mkdir(parents=True)
    figure.write_text("figure", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "figures": {
                        "s11": {
                            "overview": "fig/analyses/case/figures/s11/overview.png"
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert cached_manifest_figures(manifest_path) == {
        "s11": {"overview": figure}
    }


def test_cached_manifest_figures_recovers_images_when_manifest_entries_are_empty(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "analysis"
    figure = output_dir / "figures" / "polar" / "overview.png"
    sidecar = output_dir / "figures" / "polar" / "overview.csv"
    figure.parent.mkdir(parents=True)
    figure.write_text("figure", encoding="utf-8")
    sidecar.write_text("value\n1\n", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"outputs": {"figures": {}}}),
        encoding="utf-8",
    )

    assert cached_manifest_figures(manifest_path) == {
        "polar": {"overview": figure}
    }


def test_write_manifest_serializes_named_table_contract(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_manifest(
        path,
        sparameter_path=tmp_path / "case.z1p",
        dispersion_path=tmp_path / "dispersion",
        marker_role="sim",
        modes=("z11_impedance",),
        detection={},
        tables={},
        figures={},
        table_contract="z11",
        table_schema_version=2,
        table_constants={"z11_pts": {"z_name": "Z11"}},
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["table_contract"] == "z11"
    assert manifest["table_schema_version"] == 2
