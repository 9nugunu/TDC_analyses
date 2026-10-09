"""Assemble CST-Run-Analyze batch exports into one navigator-indexed sim dataset.

A delivery copies selected two-port exports from one or more verified batch
folders, puts the input port on Touchstone port 1, and names each file
``<stem>_<run id>.s2p`` so the existing navigator-based loaders can read it.
Run IDs and unchanged navigator columns come from a reference navigator; the
varying parameter comes from each run's ``result.json`` and is cross-checked
against the CST parameter header. Nothing is recomputed or renormalized.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from deflector_tuning.data_loading.loaders.sim_loader import _read_cst_parameter_header
from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.data_loading.touchstone_ports import write_s2p_with_ports_swapped

MANIFEST_NAME = "MANIFEST.sha256"
NAVIGATOR_NAME = "result_navigator.csv"
SOURCE_MAP_NAME = "source_map.csv"
README_NAME = "README.md"
NAVIGATOR_RUN_ID_COLUMN = " 3D Run ID"
INPUT_PORT_ON_TOUCHSTONE_PORT_1 = 1
# Touchstone v1 two-port rows are S11, S21, S12, S22; a port swap reverses them.
SWAPPED_TWO_PORT_COLUMNS = [3, 2, 1, 0]
PORT_ASSIGNMENT = re.compile(r"!\s*Touchstone port\s+\d+\s*=")
RAW_PORT_ASSIGNMENTS = ['! Touchstone port 1 = CST MWS port 1 ("")',
                        '! Touchstone port 2 = CST MWS port 2 ("")']


@dataclass(frozen=True)
class BatchRun:
    """One completed run of a CST-Run-Analyze batch, with verified digests."""

    batch: str
    batch_dir: Path
    label: str
    parameters: dict[str, str]
    export_path: Path
    export_sha256: str
    result_path: Path
    result_sha256: str
    result: dict[str, Any]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sha256_manifest(path: Path) -> dict[str, str]:
    """Read ``<sha256>  <relative path>`` lines into {relative path: digest}."""
    entries: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            entries[relative.strip()] = digest.strip()
    return entries


def verify_batch_manifest(batch_dir: Path) -> dict[str, str]:
    """Check every manifest entry of a batch folder; return the verified digests."""
    manifest = read_sha256_manifest(Path(batch_dir) / MANIFEST_NAME)
    if not manifest:
        raise ValueError(f"Empty manifest: {batch_dir}")
    for relative, expected in manifest.items():
        actual = sha256_of(Path(batch_dir) / relative)
        if actual != expected:
            raise ValueError(f"Manifest hash mismatch in {Path(batch_dir).name}: {relative}")
    return manifest


def select_batch_runs(batch_dir: Path, label_pattern: str) -> list[BatchRun]:
    """Return completed runs whose label matches, verified against the batch manifest."""
    batch_dir = Path(batch_dir)
    manifest = verify_batch_manifest(batch_dir)
    spec = json.loads((batch_dir / "spec.json").read_text(encoding="utf-8"))
    if spec["batch"] != batch_dir.name:
        raise ValueError(f"spec.json batch {spec['batch']!r} does not name folder {batch_dir.name}")
    spec_runs = {run["label"]: run for run in spec["runs"]}
    with (batch_dir / NAVIGATOR_NAME).open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if re.search(label_pattern, row["label"])]
    if not rows:
        raise ValueError(f"No run label in {batch_dir.name} matches {label_pattern!r}")
    runs = []
    for row in rows:
        if row["status"] != "complete":
            raise ValueError(f"{batch_dir.name} {row['label']}: status {row['status']}")
        spec_run = spec_runs[row["label"]]
        result_relative = f"{spec_run['folder']}/result.json"
        result = json.loads((batch_dir / result_relative).read_text(encoding="utf-8"))
        export_relative = result["export"]
        if (result["status"] != "complete" or export_relative != row["export"]
                or result["run"]["parameters"] != spec_run["parameters"]):
            raise ValueError(f"{batch_dir.name} {row['label']}: result.json disagrees with spec/navigator")
        for relative in (export_relative, result_relative):
            if relative not in manifest:
                raise ValueError(f"{batch_dir.name}: {relative} is not covered by {MANIFEST_NAME}")
        if manifest[export_relative] != result["export_sha256"]:
            raise ValueError(f"{batch_dir.name} {row['label']}: export digest differs from result.json")
        runs.append(BatchRun(batch=batch_dir.name, batch_dir=batch_dir, label=row["label"],
                             parameters=dict(spec_run["parameters"]),
                             export_path=batch_dir / export_relative,
                             export_sha256=manifest[export_relative],
                             result_path=batch_dir / result_relative,
                             result_sha256=manifest[result_relative], result=result))
    return runs


def read_cst_navigator(path: Path) -> list[dict[str, str]]:
    """Read a CST result navigator (tab-separated, quoted) keeping column names verbatim."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or NAVIGATOR_RUN_ID_COLUMN not in rows[0]:
        raise ValueError(f"Not a CST result navigator: {path}")
    return rows


def write_cst_navigator(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write the CST navigator layout: every field quoted, tab separated, CRLF lines."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow(columns)
        writer.writerows([[row[column] for column in columns] for row in rows])


def reference_run_id(reference_rows: list[dict[str, str]], header: dict[str, Any]) -> dict[str, str]:
    """Find the one reference navigator row whose parameters equal the CST header values."""
    names = [name for name in reference_rows[0] if name != NAVIGATOR_RUN_ID_COLUMN]
    matches = [row for row in reference_rows
               if all(row[name] != "" and float(row[name]) == float(header[name]) for name in names)]
    if len(matches) != 1:
        raise ValueError(f"Expected one reference navigator row for {[header[n] for n in names]}, "
                         f"found {len(matches)}")
    return matches[0]


def _numeric_values(path: Path) -> tuple[list[float], list[list[complex]], Any]:
    parsed = read_touchstone(path)
    return parsed.frequency, parsed.values, parsed.header


def deliver_run(run: BatchRun, input_port: int, target: Path) -> dict[str, Any]:
    """Write one export with the input port on Touchstone port 1 and prove the mapping exact."""
    if input_port == INPUT_PORT_ON_TOUCHSTONE_PORT_1:
        shutil.copyfile(run.export_path, target)
        permutation = [0, 1, 2, 3]
        transform = "copy"
    elif input_port == 2:
        write_s2p_with_ports_swapped(run.export_path, target)
        permutation = SWAPPED_TWO_PORT_COLUMNS
        transform = "swap_ports_1_2"
    else:
        raise ValueError(f"Unsupported input port index {input_port}")
    source_frequency, source_values, source_header = _numeric_values(run.export_path)
    frequency, values, header = _numeric_values(target)
    reordered = [[row[index] for index in permutation] for row in source_values]
    if header != source_header or frequency != source_frequency or values != reordered:
        raise ValueError(f"Delivered file does not reproduce {run.export_path.name} exactly")
    return {"port_transform": transform, "column_permutation_zero_based": permutation,
            "sample_count": len(frequency), "frequency_min_GHz": frequency[0],
            "frequency_max_GHz": frequency[-1]}


def _header_comments(path: Path) -> list[str]:
    comments = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                break
            if line.startswith("!"):
                comments.append(line.rstrip("\n"))
    return comments


def assemble_delivery(root: Path, config: dict[str, Any], config_name: str, *,
                      overwrite: bool = False) -> dict[str, Any]:
    """Build the delivery folder described by ``config``; return the source map rows.

    ``config_name`` is recorded in the generated README as the provenance of the folder.
    """
    root = Path(root)
    output = root / config["output_dataset"]
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Delivery folder is not empty: {output}")
    reference_path = root / config["reference_navigator"]
    reference_rows = read_cst_navigator(reference_path)
    columns = list(reference_rows[0])
    varying = config["varying_parameter"]
    output.mkdir(parents=True, exist_ok=True)
    delivered: list[dict[str, Any]] = []
    for batch in config["batches"]:
        batch_dir = root / batch["dataset"]
        spec = json.loads((batch_dir / "spec.json").read_text(encoding="utf-8"))
        expected_count = int(spec["frequency_GHz"][2])
        for run in select_batch_runs(batch_dir, batch["label_pattern"]):
            header = {name.removeprefix("sim_"): value
                      for name, value in _read_cst_parameter_header(run.export_path).items()}
            if float(header[varying]) != float(run.parameters[varying]):
                raise ValueError(f"{run.label}: CST header {varying} differs from result.json")
            comments = _header_comments(run.export_path)
            if [line for line in comments if PORT_ASSIGNMENT.match(line)] != RAW_PORT_ASSIGNMENTS:
                raise ValueError(f"{run.label}: unexpected raw port assignment")
            reference = reference_run_id(reference_rows, header)
            run_id = reference[NAVIGATOR_RUN_ID_COLUMN]
            target = output / f"{config['file_stem']}_{int(run_id)}.s2p"
            if target.name in {item["delivered_file"] for item in delivered}:
                raise ValueError(f"Two runs map to {target.name}")
            mapping = deliver_run(run, batch["input_port_touchstone_index"], target)
            if mapping["sample_count"] != expected_count:
                raise ValueError(f"{run.label}: {mapping['sample_count']} samples, spec says {expected_count}")
            navigator_row = {**reference, varying: run.parameters[varying]}
            delivered.append({
                "run_id": int(run_id), "delivered_file": target.name, "delivered_sha256": sha256_of(target),
                **{name: navigator_row[name] for name in columns if name != NAVIGATOR_RUN_ID_COLUMN},
                "source_batch": run.batch, "source_run_label": run.label,
                "source_file": f"{batch['dataset']}/{run.export_path.name}",
                "source_sha256": run.export_sha256,
                "source_result_json": f"{batch['dataset']}/{run.result_path.relative_to(batch_dir).as_posix()}",
                "source_result_sha256": run.result_sha256,
                "seed_sha256": run.result["seed_sha256"],
                "final_mesh_cells": run.result["final_mesh_cells"],
                "adaptive_mesh_converged": run.result["adaptive_mesh_converged"],
                "broadband_converged": run.result["broadband_converged"],
                **mapping, "column_permutation_zero_based": " ".join(map(str, mapping["column_permutation_zero_based"])),
                "_navigator_row": navigator_row,
            })
    delivered.sort(key=lambda item: item["run_id"])
    expected = sorted(float(value) for value in config["expected_values"])
    found = sorted(float(item[varying]) for item in delivered)
    if found != expected:
        raise ValueError(f"Delivered {varying} values {found} differ from configured {expected}")
    write_cst_navigator(output / NAVIGATOR_NAME, columns, [item.pop("_navigator_row") for item in delivered])
    source_map = output / SOURCE_MAP_NAME
    with source_map.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(delivered[0]))
        writer.writeheader()
        writer.writerows(delivered)
    # LF line ends, as in the CST-Run-Analyze batch manifests (sha256sum -c compatible).
    (output / README_NAME).write_text(_readme(config, config_name, delivered), encoding="utf-8", newline="\n")
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != MANIFEST_NAME)
    (output / MANIFEST_NAME).write_text(
        "".join(f"{sha256_of(path)}  {path.name}\n" for path in files), encoding="utf-8", newline="\n")
    return {"output": output, "rows": delivered, "file_count": len(files)}


def _readme(config: dict[str, Any], config_name: str, rows: list[dict[str, Any]]) -> str:
    batches = "\n".join(f"- `{batch['dataset']}`: labels matching `{batch['label_pattern']}`, "
                        f"input port = Touchstone port {batch['input_port_touchstone_index']}"
                        for batch in config["batches"])
    table = "\n".join(f"| {row['run_id']} | {row[config['varying_parameter']]} | {row['source_batch']} | "
                      f"{row['source_run_label']} | {row['port_transform']} | {row['final_mesh_cells']} |"
                      for row in rows)
    return (f"# {Path(config['output_dataset']).name}\n\n{config['description']}\n\n"
            "Generated by `scripts/build_cst_batch_delivery.py` from "
            f"`{config_name}`. Do not edit by hand.\n\n"
            f"## Sources (each verified against its `{MANIFEST_NAME}`)\n\n{batches}\n\n"
            "## Conventions\n\n"
            "- Touchstone port 1 = CST MWS port 2 = input port in every delivered file "
            "(raw exports had it on Touchstone port 2 and were port-swapped with "
            "`deflector_tuning.data_loading.touchstone_ports`). Values are unchanged; the swap is "
            "checked to reproduce the raw numbers exactly. No renormalization (`# GHz S RI R 0`).\n"
            f"- File names are `{config['file_stem']}_<run id>.s2p`. Run IDs and the unchanged "
            f"navigator columns are taken from `{config['reference_navigator']}` "
            f"(same {config['varying_parameter']} and NumTune -> same run ID); "
            f"{config['varying_parameter']} comes from each run's `result.json`.\n"
            f"- `{SOURCE_MAP_NAME}` maps every delivered file to its batch export, digests and mesh status.\n\n"
            f"## Runs\n\n| Run ID | {config['varying_parameter']} | Batch | Label | Transform | Final mesh cells |\n"
            f"|---|---|---|---|---|---|\n{table}\n")
