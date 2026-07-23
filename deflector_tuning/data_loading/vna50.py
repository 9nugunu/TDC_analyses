"""Create VNA-reference reflection data from physical input impedance."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.dataset_naming import validate_dataset_id
from deflector_tuning.data_loading.source_layer import DataLayer


_TRANSFORM_NAME = "z11_to_vna_reflection"


@dataclass(frozen=True)
class Vna50Export:
    """Paths written by a VNA-reference Touchstone transformation."""

    dataset_path: Path
    touchstone_files: tuple[Path, ...]
    manifest_path: Path


def impedance_to_reflection(
    z_input: object,
    reference_ohm: float = 50.0,
) -> np.ndarray:
    """Convert complex input impedance to a reflection coefficient.

    Parameters
    ----------
    z_input:
        Complex physical input impedance in ohms.
    reference_ohm:
        Positive finite real VNA reference impedance in ohms.

    Returns
    -------
    numpy.ndarray
        ``(z_input - reference_ohm) / (z_input + reference_ohm)``.
    """

    z0 = float(reference_ohm)
    if not np.isfinite(z0) or z0 <= 0.0:
        raise ValueError("reference_ohm must be positive and finite")

    impedance = np.asarray(z_input, dtype=complex)
    if not np.all(np.isfinite(impedance.real) & np.isfinite(impedance.imag)):
        raise ValueError("Z input contains non-finite values")

    denominator = impedance + z0
    singular_tolerance = np.finfo(float).eps * max(1.0, z0) * 16.0
    if np.any(np.abs(denominator) <= singular_tolerance):
        raise ValueError("Z input produces a numerically singular denominator")

    return (impedance - z0) / denominator


def export_vna50_touchstone_dataset(
    source_path: str | Path,
    output_path: str | Path,
    reference_ohm: float = 50.0,
) -> Vna50Export:
    """Export a one-port Z11 simulation dataset as reference-impedance S1P."""

    source = Path(source_path)
    output = Path(output_path)
    validate_dataset_id(output.name, DataLayer.SIM)
    z_table = DataLoader().load(source)
    _validate_z11_table(z_table)

    grouped = _validated_source_groups(z_table)
    _validate_navigator(source, grouped)
    frequency = _shared_frequency_grid(grouped)
    file_records, output_values = _build_output_values(grouped, reference_ohm)
    expected_mapping = [
        {"source": record["source"], "output": record["output"]}
        for record in file_records
    ]
    _validate_existing_output(
        output,
        source_dataset=source.name,
        reference_ohm=float(reference_ohm),
        expected_mapping=expected_mapping,
    )

    output.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    for record, values in zip(file_records, output_values, strict=True):
        destination = output / str(record["output"])
        _write_s1p(
            destination,
            frequency=frequency,
            reflection=values,
            source_dataset=source.name,
            reference_ohm=float(reference_ohm),
        )
        output_files.append(destination)

    navigator_source = source / "result_navigator.csv"
    (output / "result_navigator.csv").write_bytes(navigator_source.read_bytes())

    maximum_abs_gamma = max(
        float(np.max(np.abs(values)))
        for values in output_values
    )
    manifest = {
        "transform": _TRANSFORM_NAME,
        "source_dataset": source.name,
        "output_dataset": output.name,
        "formula": "Gamma = (Zin - Z0) / (Zin + Z0)",
        "reference_ohm": float(reference_ohm),
        "input_parameter": "Z11",
        "output_parameter": "S11",
        "files": file_records,
        "frequency_min_ghz": float(frequency[0]),
        "frequency_max_ghz": float(frequency[-1]),
        "point_count_per_file": int(len(frequency)),
        "maximum_abs_gamma": maximum_abs_gamma,
    }
    manifest_path = output / "transform_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Vna50Export(
        dataset_path=output,
        touchstone_files=tuple(output_files),
        manifest_path=manifest_path,
    )


def _validate_z11_table(table: pd.DataFrame) -> None:
    required = {
        "source_file",
        "run_id",
        "freq_ghz",
        "z_name",
        "z_real_ohm",
        "z_imag_ohm",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Expected one-port Z11 input; missing columns: {sorted(missing)}"
        )
    if set(table["z_name"].dropna()) != {"Z11"}:
        raise ValueError("VNA-reference export requires one-port Z11 input")


def _validated_source_groups(
    table: pd.DataFrame,
) -> list[tuple[str, int, pd.DataFrame]]:
    groups: list[tuple[str, int, pd.DataFrame]] = []
    seen_run_ids: set[int] = set()
    for source_file, file_table in table.groupby("source_file", sort=True):
        run_ids = pd.to_numeric(file_table["run_id"], errors="coerce").dropna().unique()
        if len(run_ids) == 0:
            raise ValueError(f"Source file {source_file} has a missing run ID")
        if len(run_ids) != 1:
            raise ValueError(f"Source file {source_file} has multiple run IDs")
        run_id = int(run_ids[0])
        if run_id in seen_run_ids:
            raise ValueError(f"Source files have a duplicated run ID: {run_id}")
        seen_run_ids.add(run_id)
        ordered = file_table.sort_values("freq_ghz", kind="mergesort").reset_index(drop=True)
        if ordered["freq_ghz"].duplicated().any():
            raise ValueError(f"Source file {source_file} contains duplicate frequencies")
        groups.append((str(source_file), run_id, ordered))
    if not groups:
        raise ValueError("VNA-reference export requires at least one Z11 source file")
    return groups


def _validate_navigator(
    source: Path,
    groups: list[tuple[str, int, pd.DataFrame]],
) -> None:
    navigator_path = source / "result_navigator.csv"
    if not navigator_path.exists():
        raise ValueError(f"Missing result_navigator.csv in {source}")
    navigator = pd.read_csv(navigator_path, sep="\t")
    navigator = navigator.rename(columns=lambda name: str(name).strip().strip('"'))
    if "3D Run ID" not in navigator:
        raise ValueError("result_navigator.csv is missing the 3D Run ID column")
    metadata_columns = [column for column in navigator.columns if column != "3D Run ID"]
    if metadata_columns:
        has_metadata = navigator[metadata_columns].notna().any(axis=1)
        navigator = navigator.loc[has_metadata]
    run_ids = pd.to_numeric(navigator["3D Run ID"], errors="coerce").dropna().astype(int)
    if run_ids.duplicated().any():
        raise ValueError("result_navigator.csv contains duplicated run IDs")
    file_run_ids = {run_id for _, run_id, _ in groups}
    navigator_run_ids = set(run_ids.tolist())
    if navigator_run_ids != file_run_ids:
        raise ValueError(
            "result_navigator.csv does not map one-to-one to source files: "
            f"files={sorted(file_run_ids)}, navigator={sorted(navigator_run_ids)}"
        )


def _shared_frequency_grid(
    groups: list[tuple[str, int, pd.DataFrame]],
) -> np.ndarray:
    reference = groups[0][2]["freq_ghz"].to_numpy(dtype=float)
    if len(reference) == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("Source frequency grid must be finite and non-empty")
    for source_file, _, file_table in groups[1:]:
        candidate = file_table["freq_ghz"].to_numpy(dtype=float)
        if not np.array_equal(candidate, reference):
            raise ValueError(
                f"Source files use different frequency grids; mismatch in {source_file}"
            )
    return reference


def _build_output_values(
    groups: list[tuple[str, int, pd.DataFrame]],
    reference_ohm: float,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    records: list[dict[str, object]] = []
    values: list[np.ndarray] = []
    for source_file, run_id, file_table in groups:
        impedance = (
            file_table["z_real_ohm"].to_numpy(dtype=float)
            + 1j * file_table["z_imag_ohm"].to_numpy(dtype=float)
        )
        reflection = impedance_to_reflection(impedance, reference_ohm)
        output_file = Path(source_file).with_suffix(".s1p").name
        records.append(
            {
                "source": source_file,
                "output": output_file,
                "run_id": run_id,
                "point_count": int(len(file_table)),
                "frequency_min_ghz": float(file_table["freq_ghz"].iloc[0]),
                "frequency_max_ghz": float(file_table["freq_ghz"].iloc[-1]),
                "maximum_abs_gamma": float(np.max(np.abs(reflection))),
            }
        )
        values.append(reflection)
    return records, values


def _validate_existing_output(
    output: Path,
    *,
    source_dataset: str,
    reference_ohm: float,
    expected_mapping: list[dict[str, str]],
) -> None:
    if not output.exists():
        return
    manifest_path = output / "transform_manifest.json"
    existing_s1p = sorted(path.name for path in output.glob("*.s1p"))
    if not manifest_path.exists():
        if existing_s1p or any(output.iterdir()):
            raise ValueError(f"Refusing conflicting output dataset without owned manifest: {output}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid transform manifest in conflicting output dataset: {output}") from error
    existing_mapping = [
        {"source": record.get("source"), "output": record.get("output")}
        for record in manifest.get("files", [])
    ]
    expected_outputs = sorted(record["output"] for record in expected_mapping)
    identity_matches = (
        manifest.get("transform") == _TRANSFORM_NAME
        and manifest.get("source_dataset") == source_dataset
        and manifest.get("output_dataset") == output.name
        and manifest.get("reference_ohm") == reference_ohm
        and existing_mapping == expected_mapping
        and existing_s1p == expected_outputs
    )
    if not identity_matches:
        raise ValueError(f"Refusing conflicting output dataset: {output}")


def _write_s1p(
    path: Path,
    *,
    frequency: np.ndarray,
    reflection: np.ndarray,
    source_dataset: str,
    reference_ohm: float,
) -> None:
    formula_label = "Gamma50" if reference_ohm == 50.0 else "Gamma"
    lines = [
        "! Derived from CST Z11",
        f"! {formula_label} = (Zin - {reference_ohm:.17g}) / (Zin + {reference_ohm:.17g})",
        f"! Source dataset: {source_dataset}",
        f"# GHz S RI R {reference_ohm:.17g}",
    ]
    lines.extend(
        f"{freq:.17g} {value.real:.17g} {value.imag:.17g}"
        for freq, value in zip(frequency, reflection, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
