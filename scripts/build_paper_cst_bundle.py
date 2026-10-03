"""Export source-auditable CST short-state figures and paired numerical tables.

Only configured sources inside data/sim are accepted. All complex values come
from the existing Touchstone reader with no impedance conversion or
renormalization. Metadata selection, validation, and plotting are generic over
the configured cases; file indices are not used as physical state definitions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.data_loading.loaders.sim_loader import _read_cst_parameter_header
from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.visualization.lps_paper_style import apply_lps_style, finish_axes, save_paper_figure


def sha256(path: Path) -> str:
    """Hash the source bytes, independently of parsing or text newlines."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_sample_index(frequency_GHz: np.ndarray, requested_GHz: float) -> int:
    """Choose the nearest exported sample; an exact tie selects the first row."""
    frequencies = np.asarray(frequency_GHz, dtype=float)
    if (frequencies.ndim != 1 or frequencies.size == 0
            or not np.isfinite(frequencies).all() or not np.isfinite(requested_GHz)
            or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("Frequency grid must be finite, nonempty and strictly increasing")
    if not frequencies[0] <= requested_GHz <= frequencies[-1]:
        raise ValueError(f"Marker {requested_GHz} GHz lies outside the exported grid")
    return int(np.argmin(np.abs(frequencies - requested_GHz)))


def phase_pair_metrics(first: complex, second: complex, target_deg: float) -> dict[str, float]:
    """Return directed second/first phase separation and wrapped target residual.

    Separation is in [0, 360); target residual is in [-180, 180). This is a
    reference-phase comparison, not an independent impedance matching test.
    """
    if not all(np.isfinite(value) and abs(value) > 0 for value in (first, second)):
        raise ValueError("Phase inputs must be finite and nonzero")
    delta = float(np.rad2deg(np.angle(second / first)) % 360.0)
    # Floating point modulo can round an infinitesimal negative value to 360.
    if delta >= 360.0:
        delta = 0.0
    return {
        "first_real": float(first.real), "first_imag": float(first.imag),
        "second_real": float(second.real), "second_imag": float(second.imag),
        "first_phase_deg": float(np.rad2deg(np.angle(first))),
        "second_phase_deg": float(np.rad2deg(np.angle(second))),
        "first_modulus": float(abs(first)), "second_modulus": float(abs(second)),
        "delta_deg": delta,
        "target_delta_deg": float(target_deg),
        "signed_target_residual_deg": float((delta - target_deg + 180.0) % 360.0 - 180.0),
        "relative_phase_psi_deg": float((delta - 180.0 + 180.0) % 360.0 - 180.0),
    }


def validate_pair_metadata(
    first: dict[str, Any], second: dict[str, Any], short_state_fields: list[str],
    *, require_same_project: bool = True,
) -> dict[str, Any]:
    """Reject any metadata difference not explicitly declared as a state field."""
    mismatches: dict[str, Any] = {}
    fields = ["option_line", "port_assignments"]
    if require_same_project:
        fields.append("project")
    for name in fields:
        if first[name] != second[name]:
            mismatches[name] = {"first": first[name], "second": second[name]}
    parameters_a, parameters_b = first["parameters"], second["parameters"]
    varying: dict[str, Any] = {}
    missing = "<missing>"
    for name in sorted(set(parameters_a) | set(parameters_b)):
        left, right = parameters_a.get(name, missing), parameters_b.get(name, missing)
        if left != right:
            difference = {"first": left, "second": right}
            if name in short_state_fields and left != missing and right != missing:
                varying[name] = difference
            else:
                mismatches[name] = difference
    if mismatches:
        raise ValueError("Pair metadata mismatch: " + json.dumps(mismatches, sort_keys=True))
    return {"consistent": True, "varying_short_state_fields": varying,
            "different_project_names": ([first["project"], second["project"]]
                                        if first["project"] != second["project"] else [])}


def radius_phase_response(
    radii: np.ndarray, raw_phases_deg: np.ndarray, nominal_radius: float
) -> dict[str, Any]:
    """Unwrap along radius and calculate the centered local phase derivative."""
    radii, phases = np.asarray(radii, dtype=float), np.asarray(raw_phases_deg, dtype=float)
    if (radii.ndim != 1 or radii.shape != phases.shape or len(radii) < 3
            or not np.isfinite(radii).all() or not np.isfinite(phases).all()
            or np.any(np.diff(radii) <= 0)):
        raise ValueError("Radius and phase must be finite equal-length arrays on a sorted grid")
    matches = np.flatnonzero(np.isclose(radii, nominal_radius, atol=1e-10, rtol=0))
    if len(matches) != 1 or matches[0] in (0, len(radii) - 1):
        raise ValueError("Nominal radius must identify one interior grid point")
    index = int(matches[0])
    if not np.isclose(radii[index] - radii[index - 1], radii[index + 1] - radii[index],
                      atol=1e-10, rtol=0):
        raise ValueError("Nominal centered derivative requires equal adjacent radius steps")
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phases)))
    unwrapped -= 360.0 * round((unwrapped[index] - phases[index]) / 360.0)
    slope = (unwrapped[index + 1] - unwrapped[index - 1]) / (radii[index + 1] - radii[index - 1])
    return {"unwrapped_phase_deg": unwrapped.tolist(),
            "branch_turns": np.rint((unwrapped - phases) / 360.0).astype(int).tolist(),
            "nominal_index": index, "nominal_central_slope_deg_per_mm": float(slope),
            "slope_lower_radius_mm": float(radii[index - 1]),
            "slope_upper_radius_mm": float(radii[index + 1])}


def read_cst_metadata(path: Path) -> dict[str, Any]:
    """Read CST comments and reuse the existing CST parameter-header parser."""
    comments: list[str] = []
    option_line = ""
    with path.open(encoding="utf-8", errors="strict") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("!"):
                comments.append(line)
            elif line.startswith("#"):
                option_line = line
                break
    if not any("generated by CST" in line for line in comments):
        raise ValueError(f"No CST simulation export identification: {path}")
    project = next((line.split(":", 1)[1].strip() for line in comments
                    if line.startswith("! Project name:")), None)
    assignments = [line for line in comments if re.match(r"!\s*Touchstone port\s+\d+\s*=", line)]
    parameters = {key.removeprefix("sim_"): value
                  for key, value in _read_cst_parameter_header(path).items()}
    if not project or not option_line or not assignments or "NumDepth" not in parameters:
        raise ValueError(f"Incomplete CST source identity or short-state metadata: {path}")
    return {"project": project, "option_line": option_line, "parameters": parameters,
            "port_assignments": assignments, "header_comments": comments}


def simulation_path(root: Path, relative: str) -> Path:
    """Keep the initial data-kind routing explicit and reject experimental paths."""
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "data" / "sim").resolve()):
        raise ValueError(f"Only simulation data/sim paths are permitted: {relative}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_markers(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify configured markers against the actual dispersion-summary row."""
    source = root / config["marker_source"]
    with source.open(newline="", encoding="utf-8") as stream:
        candidates = [row for row in csv.DictReader(stream)
                      if int(row["mode_index"]) == config["marker_mode_index"]]
    if len(candidates) != 1:
        raise ValueError("Dispersion mode must identify exactly one source row")
    row = candidates[0]
    markers = []
    for item in config["markers"]:
        source_values = [float(row[column]) for column in item.get(
            "mean_of_columns", [item.get("source_column")])]
        derived = float(np.mean(source_values))
        if not np.isclose(derived, item["requested_GHz"], atol=1e-12, rtol=0):
            raise ValueError(f"Configured marker disagrees with dispersion source: {item['name']}")
        markers.append({**item, "source_derived_GHz": derived})
    return markers, {"path": source.relative_to(root).as_posix(), "sha256": sha256(source),
                     "mode_index": config["marker_mode_index"], "source_row": row,
                     "markers": markers}


def load_case(root: Path, case: dict[str, Any], depths: list[float]) -> tuple[dict[float, dict], dict]:
    """Select unique metadata-defined states and read raw exported S parameters."""
    folder = simulation_path(root, case["dataset"])
    candidates = sorted(path for path in folder.iterdir() if re.fullmatch(r"\.s\d+p", path.suffix.lower()))
    selected: dict[float, list[tuple[Path, dict]]] = {depth: [] for depth in depths}
    excluded = []
    for path in candidates:
        metadata = read_cst_metadata(path)
        depth = float(metadata["parameters"]["NumDepth"])
        if depth in selected:
            selected[depth].append((path, metadata))
        else:
            excluded.append({"path": path.relative_to(root).as_posix(), "NumDepth": depth})
    states: dict[float, dict] = {}
    for depth, entries in selected.items():
        if len(entries) != 1:
            raise ValueError(f"{case['id']} NumDepth={depth}: expected one source, found {len(entries)}")
        path, metadata = entries[0]
        for name, expected in case["expected_parameters"].items():
            if metadata["parameters"].get(name) != expected:
                raise ValueError(f"{path.name}: {name} does not equal configured {expected}")
        parsed = read_touchstone(path)
        if parsed.header.parameter != "S":
            raise ValueError(f"Only raw S parameters are supported: {path}")
        scales = {"Hz": 1e-9, "kHz": 1e-6, "MHz": 1e-3, "GHz": 1.0}
        frequency = np.asarray(parsed.frequency) * scales[parsed.header.frequency_unit]
        values = np.asarray(parsed.values, dtype=complex)
        # Nearest-sample validation also rejects corrupt/non-monotonic grids.
        nearest_sample_index(frequency, float(frequency[0]))
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite raw S-parameter samples: {path}")
        identity = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
                    "metadata": metadata, "touchstone_header": asdict(parsed.header),
                    "sample_count": len(frequency), "frequency_min_GHz": float(frequency[0]),
                    "frequency_max_GHz": float(frequency[-1]),
                    "frequency_step_min_GHz": float(np.min(np.diff(frequency))),
                    "frequency_step_max_GHz": float(np.max(np.diff(frequency))),
                    "exported_s_index": case["exported_s_index"],
                    "exported_port_1_assignment": metadata["port_assignments"][0]}
        if "port_transform" in case:
            transform = case["port_transform"]
            original_path = simulation_path(root, transform["original_dataset"]) / path.name
            original = read_touchstone(original_path)
            reordered = np.asarray(original.values)[:, transform["column_permutation"]]
            exact = (original.header == parsed.header
                     and np.array_equal(np.asarray(original.frequency), np.asarray(parsed.frequency))
                     and np.array_equal(reordered, values))
            if not exact:
                raise ValueError(f"Configured port reorder does not reproduce {path}")
            identity["port_transform"] = {
                "original_path": original_path.relative_to(root).as_posix(),
                "original_sha256": sha256(original_path),
                "column_permutation_zero_based": transform["column_permutation"],
                "exact_numeric_equality": True,
                "max_abs_complex_difference": float(np.max(np.abs(reordered - values))),
                "interpretation": "Exact exported port reordering; not an independent CST simulation",
            }
        states[depth] = {"frequency_GHz": frequency, "s": values[:, case["exported_s_index"]],
                         "identity": identity}
    audit = {"case": case, "candidate_count": len(candidates), "selected_count": len(states),
             "excluded_states": excluded, "sources": [states[depth]["identity"] for depth in depths]}
    return states, audit


def make_tables(root: Path, config: dict[str, Any]) -> tuple[list[dict], list[dict], dict]:
    """Build paired metrics, per-state geometry rows, and the source manifest."""
    if config["data_kind"] != "simulation":
        raise ValueError("This publication bundle accepts simulation sources only")
    markers, marker_audit = resolve_markers(root, config)
    depths = sorted({float(pair[field]) for pair in config["pairs"]
                     for field in ("first_num_depth", "second_num_depth")})
    metrics, geometry, case_audits = [], [], []
    for case in config["cases"]:
        states, audit = load_case(root, case, depths)
        checks = []
        for depth, state in states.items():
            identity = state["identity"]
            metadata = identity["metadata"]
            geometry.append({"case": case["id"], "data_kind": "simulation",
                             "source_file": identity["path"], "sha256": identity["sha256"],
                             "CST_project": metadata["project"], "option_line": metadata["option_line"],
                             "exported_port_1_assignment": identity["exported_port_1_assignment"],
                             "sample_count": identity["sample_count"],
                             **metadata["parameters"]})
        for pair in config["pairs"]:
            first = states[float(pair["first_num_depth"])]
            second = states[float(pair["second_num_depth"])]
            identities = [first["identity"], second["identity"]]
            check = validate_pair_metadata(identities[0]["metadata"], identities[1]["metadata"],
                                           config["short_state_fields"])
            if not np.array_equal(first["frequency_GHz"], second["frequency_GHz"]):
                raise ValueError(f"Frequency-grid mismatch within {case['id']}/{pair['name']}")
            checks.append({"pair": pair["name"], **check, "identical_frequency_grids": True})
            for marker in markers:
                indices = [nearest_sample_index(state["frequency_GHz"], marker["requested_GHz"])
                           for state in (first, second)]
                actual = [float(state["frequency_GHz"][index])
                          for state, index in zip((first, second), indices)]
                row = {"case": case["id"], "data_kind": "simulation", "pair": pair["name"],
                       "first_label": pair["first_label"], "second_label": pair["second_label"],
                       "first_NumDepth": pair["first_num_depth"], "second_NumDepth": pair["second_num_depth"],
                       "marker": marker["name"], "requested_frequency_GHz": marker["requested_GHz"],
                       "first_sample_index_zero_based": indices[0], "second_sample_index_zero_based": indices[1],
                       "first_actual_frequency_GHz": actual[0], "second_actual_frequency_GHz": actual[1],
                       "first_frequency_error_kHz": (actual[0] - marker["requested_GHz"]) * 1e6,
                       "second_frequency_error_kHz": (actual[1] - marker["requested_GHz"]) * 1e6,
                       "first_source": identities[0]["path"], "second_source": identities[1]["path"],
                       "first_sha256": identities[0]["sha256"], "second_sha256": identities[1]["sha256"],
                       "option_line": identities[0]["metadata"]["option_line"],
                       "exported_port_1_assignment": identities[0]["exported_port_1_assignment"],
                       "target_relative_phase_psi_deg": marker["target_delta_deg"] - 180,
                       "metadata_pair_consistent": True,
                       **phase_pair_metrics(first["s"][indices[0]], second["s"][indices[1]],
                                            marker["target_delta_deg"])}
                metrics.append(row)
        # All four selected states also share the same non-state metadata.
        for depth in depths[1:]:
            validate_pair_metadata(states[depths[0]]["identity"]["metadata"],
                                   states[depth]["identity"]["metadata"], config["short_state_fields"])
        audit["pair_consistency_checks"] = checks
        audit["all_selected_states_non_state_metadata_consistent"] = True
        case_audits.append(audit)
    return metrics, geometry, {"marker_source": marker_audit, "cases": case_audits}


def make_radius_table(root: Path, config: dict[str, Any]) -> tuple[list[dict], dict]:
    """Read the fixed-state radius scan and prove its nominal link to the short state."""
    scan = config["radius_scan"]
    axis = scan["axis_parameter"]
    folder = simulation_path(root, scan["dataset"])
    files = sorted(path for path in folder.iterdir() if re.fullmatch(r"\.s\d+p", path.suffix.lower()))
    markers, _ = resolve_markers(root, config)
    records, sources = [], []
    baseline_metadata = None
    baseline_grid = None
    nominal_trace = None
    scales = {"Hz": 1e-9, "kHz": 1e-6, "MHz": 1e-3, "GHz": 1.0}
    for path in files:
        metadata = read_cst_metadata(path)
        for name, value in scan["expected_parameters"].items():
            if metadata["parameters"].get(name) != value:
                raise ValueError(f"Radius scan mismatch: {path.name}: expected {name}={value}")
        for name in scan.get("linked_equal_to_axis", []):
            if metadata["parameters"].get(name) != metadata["parameters"][axis]:
                raise ValueError(f"Radius scan linked parameter mismatch: {path.name}: {name} != {axis}")
        if baseline_metadata is None:
            baseline_metadata = metadata
        else:
            validate_pair_metadata(baseline_metadata, metadata, [axis, *scan.get("linked_equal_to_axis", [])])
        radius = float(metadata["parameters"][axis])
        trace = read_touchstone(path)
        if trace.header.parameter != "S" or trace.header.reference_ohm != 0:
            raise ValueError(f"Expected unrenormalized raw CST S parameters: {path}")
        frequency = np.asarray(trace.frequency) * scales[trace.header.frequency_unit]
        raw_s = np.asarray(trace.values, dtype=complex)[:, scan["exported_s_index"]]
        if baseline_grid is None:
            baseline_grid = frequency
        elif not np.array_equal(baseline_grid, frequency):
            raise ValueError(f"Frequency grid changes across radius scan: {path}")
        source = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
                  "metadata": metadata, "touchstone_header": asdict(trace.header),
                  "sample_count": len(frequency), "radius_mm": radius}
        sources.append(source)
        if np.isclose(radius, scan["nominal_radius"], atol=1e-10, rtol=0):
            nominal_trace = (frequency, raw_s, source)
        for marker in markers:
            index = nearest_sample_index(frequency, marker["requested_GHz"])
            signal = raw_s[index]
            if not np.isfinite(signal) or abs(signal) == 0:
                raise ValueError(f"Undefined phase at radius marker: {path}")
            records.append({"data_kind": "simulation", "r_c_mm": radius,
                            "coupler_path_h_mm": metadata["parameters"]["coupler_path_h"],
                            "marker": marker["name"], "requested_frequency_GHz": marker["requested_GHz"],
                            "actual_frequency_GHz": float(frequency[index]),
                            "frequency_error_kHz": float((frequency[index] - marker["requested_GHz"]) * 1e6),
                            "nearest_sample_distance_kHz": float(abs(frequency[index] - marker["requested_GHz"]) * 1e6),
                            "frequency_grid_step_GHz": float(np.median(np.diff(frequency))),
                            "sample_index_zero_based": index, "raw_real": float(signal.real),
                            "raw_imag": float(signal.imag), "raw_modulus": float(abs(signal)),
                            "raw_phase_deg": float(np.angle(signal, deg=True)),
                            "source_file": source["path"], "sha256": source["sha256"],
                            "CST_project": metadata["project"], "option_line": metadata["option_line"],
                            "exported_port_1_assignment": metadata["port_assignments"][0],
                            **metadata["parameters"]})
    grid = scan["expected_grid"]
    radii = np.array(sorted(source["radius_mm"] for source in sources))
    expected = grid["minimum"] + np.arange(grid["count"]) * grid["step"]
    if (len(radii) != grid["count"] or not np.allclose(radii, expected, atol=1e-10, rtol=0)
            or not np.isclose(radii[-1], grid["maximum"], atol=1e-10, rtol=0)):
        raise ValueError("Radius scan does not reproduce the configured count and complete uniform grid")
    responses = {}
    for marker in markers:
        rows = sorted((row for row in records if row["marker"] == marker["name"]), key=lambda row: row["r_c_mm"])
        response = radius_phase_response(radii, np.array([row["raw_phase_deg"] for row in rows]),
                                         scan["nominal_radius"])
        responses[marker["name"]] = response
        for index, row in enumerate(rows):
            row.update({"unwrapped_phase_deg": response["unwrapped_phase_deg"][index],
                        "phase_branch_turns": response["branch_turns"][index],
                        "radius_grid_step_mm": grid["step"], "radius_index_zero_based": index,
                        "nominal_radius_mm": scan["nominal_radius"],
                        "nominal_central_slope_deg_per_mm": response["nominal_central_slope_deg_per_mm"],
                        "slope_lower_radius_mm": response["slope_lower_radius_mm"],
                        "slope_upper_radius_mm": response["slope_upper_radius_mm"]})
    if nominal_trace is None:
        raise ValueError("Nominal radius trace missing")
    reference_folder = simulation_path(root, scan["nominal_reference_dataset"])
    reference_matches = []
    for path in reference_folder.glob("*.s*p"):
        metadata = read_cst_metadata(path)
        if metadata["parameters"].get("NumDepth") == scan["nominal_reference_NumDepth"]:
            reference_matches.append((path, metadata))
    if len(reference_matches) != 1:
        raise ValueError("Nominal short-state link requires one metadata-selected reference")
    reference_path, reference_metadata = reference_matches[0]
    nominal_frequency, nominal_s, nominal_source = nominal_trace
    geometry_check = validate_pair_metadata(nominal_source["metadata"], reference_metadata, [],
                                            require_same_project=False)
    reference_trace = read_touchstone(reference_path)
    reference_frequency = np.asarray(reference_trace.frequency) * scales[reference_trace.header.frequency_unit]
    reference_s = np.asarray(reference_trace.values)[:, scan["exported_s_index"]]
    common, nominal_indices, reference_indices = np.intersect1d(
        nominal_frequency, reference_frequency, assume_unique=True, return_indices=True)
    exact = np.array_equal(nominal_s[nominal_indices], reference_s[reference_indices])
    if len(common) != len(nominal_frequency):
        raise ValueError("Nominal radius frequency grid is not a subset of the short-state reference")
    complex_differences = np.abs(nominal_s[nominal_indices] - reference_s[reference_indices])
    phase_differences = np.angle(nominal_s[nominal_indices] / reference_s[reference_indices], deg=True)
    radius_step = float(np.median(np.diff(nominal_frequency)))
    reference_step = float(np.median(np.diff(reference_frequency)))
    record_by_marker = []
    for marker in markers:
        index_scan = nearest_sample_index(nominal_frequency, marker["requested_GHz"])
        index_reference = nearest_sample_index(reference_frequency, marker["requested_GHz"])
        record_by_marker.append({"marker": marker["name"],
                                 "scan_actual_frequency_GHz": float(nominal_frequency[index_scan]),
                                 "reference_actual_frequency_GHz": float(reference_frequency[index_reference]),
                                 "scan_raw_phase_deg": float(np.angle(nominal_s[index_scan], deg=True)),
                                 "reference_raw_phase_deg": float(np.angle(reference_s[index_reference], deg=True))})
    audit = {"config": scan, "source_count": len(sources), "table_row_count": len(records),
             "sources": sorted(sources, key=lambda item: item["radius_mm"]),
             "all_metadata_outside_radius_and_declared_links_consistent": True,
             "linked_geometry_verified": {name: f"{name} == {axis} for every source"
                                          for name in scan.get("linked_equal_to_axis", [])},
             "identical_scan_frequency_grids": True,
             "scan_frequency_step_GHz": radius_step, "scan_frequency_count": len(nominal_frequency),
             "phase_responses": responses,
             "nominal_reference_link": {
                 "reference_path": reference_path.relative_to(root).as_posix(),
                 "reference_sha256": sha256(reference_path), "metadata_check": geometry_check,
                 "reference_frequency_step_GHz": reference_step,
                 "scan_to_reference_frequency_step_ratio": radius_step / reference_step,
                 "common_frequency_count": len(common), "all_scan_frequencies_in_reference": True,
                 "complex_values_exactly_equal_at_common_frequencies": exact,
                 "nonidentical_common_samples": int(np.count_nonzero(complex_differences)),
                 "max_abs_complex_difference": float(np.max(complex_differences)),
                 "max_complex_difference_frequency_GHz": float(common[np.argmax(complex_differences)]),
                 "max_abs_phase_difference_deg": float(np.max(np.abs(phase_differences))),
                 "comparison_status": "Exact geometry/port metadata agreement; numerical trace differences retained and reported",
                 "nearest_marker_comparison": record_by_marker},
             "interpretation": "Raw phase response to configured radius scan (including declared linked geometry) under one exported port gauge; no optimal radius or coupling coefficient inferred"}
    return sorted(records, key=lambda row: (row["r_c_mm"], row["marker"])), audit


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the union of fields without dropping case-specific CST metadata."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(output_root: Path, config: dict[str, Any], rows: list[dict]) -> list[Path]:
    """Plot absolute phases and target residuals on matched, full-width axes."""
    style = apply_lps_style()
    colors = ["C0", "C1"]
    output_paths = []
    markers = config["markers"]
    tick_labels = [r"$f_{2\pi/3}$" + "\n" + f"{markers[0]['requested_GHz']:.6f}",
                   r"$f_m$" + "\n" + f"{markers[1]['requested_GHz']:.6f}",
                   r"$f_{\pi/2}$" + "\n" + f"{markers[2]['requested_GHz']:.6f}"]
    x = np.arange(len(markers), dtype=float)
    residual_limit = max(10, int(np.ceil(max(abs(row["signed_target_residual_deg"])
                                            for row in rows) / 10.0) * 10))
    for kind, stem in [("phase", "fig03-short-state-phases"),
                       ("residual", "fig04-cell-iris-residuals")]:
        fig, axes = plt.subplots(1, len(config["cases"]), figsize=(7.08, 3.05),
                                 sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.31, top=0.79, wspace=0.14)
        for index, (case, ax) in enumerate(zip(config["cases"], axes[0])):
            condition = case["expected_parameters"]
            ax.set_title(f"({chr(97 + index)}) {case['label']}\n"
                         f"NumCell={condition['NumCell']}; R_plunger={condition['R_plunger']}", pad=9)
            for pair_index, pair in enumerate(config["pairs"]):
                selected = [next(row for row in rows if row["case"] == case["id"]
                                 and row["pair"] == pair["name"] and row["marker"] == marker["name"])
                            for marker in markers]
                if kind == "phase":
                    for state_index, state in enumerate(("first", "second")):
                        offset = (pair_index * 2 + state_index - 1.5) * 0.075
                        ax.plot(x + offset, [row[f"{state}_phase_deg"] for row in selected],
                                linestyle="none", marker=("o", "s")[pair_index], markersize=5,
                                markerfacecolor="white" if state_index == 0 else colors[pair_index],
                                markeredgecolor=colors[pair_index], markeredgewidth=1.1,
                                label=pair[f"{state}_label"] + f" ({pair[f'{state}_num_depth']:g})")
                    ax.set_ylim(-195, 195)
                    ax.set_yticks([-180, -90, 0, 90, 180])
                else:
                    offset = (-0.08, 0.08)[pair_index]
                    ax.plot(x + offset, [row["signed_target_residual_deg"] for row in selected],
                            linestyle="none", marker=("o", "s")[pair_index], markersize=5,
                            color=colors[pair_index], label=pair["label"])
                    ax.set_ylim(-residual_limit * 1.10, residual_limit * 1.10)
            ax.set_xticks(x, tick_labels)
            ax.set_xlim(-0.35, len(markers) - 0.65)
            ax.grid(axis="y", color="0.88", lw=0.6)
            ax.axhline(0, color="0.40", lw=0.8, ls="--", zorder=0)
            ax.tick_params(labelsize=8)
            ax.set_axisbelow(True)
        axes[0, 0].set_ylabel("Raw $S_{11}$ phase (deg)" if kind == "phase"
                              else "Signed reference residual (deg)", fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.985),
                   ncol=4 if kind == "phase" else 2, frameon=False, fontsize=8,
                   handletextpad=0.4, columnspacing=1.3)
        fig.text(0.54, 0.115, "Requested marker frequency (GHz)", ha="center", fontsize=9)
        footer = ("CST simulation; nearest samples; raw exported port order; R 0 (unrenormalized)."
                  if kind == "phase" else
                  "Reference targets: 240°, 180°, 120°; cases have different geometries.")
        fig.text(0.54, 0.022, footer, ha="center", fontsize=7.5)
        for suffix in ("pdf", "png"):
            path = output_root / "fig" / f"{stem}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"Creator": "build_paper_cst_bundle.py"}
            if suffix == "pdf":
                metadata.update({"CreationDate": None, "ModDate": None})
            for ax in axes.flat:
                finish_axes(ax, scale=0.85)
            save_paper_figure(fig, path, metadata=metadata)
            output_paths.append(path)
        plt.close(fig)
    return output_paths


def draw_radius_figure(output_root: Path, config: dict[str, Any], rows: list[dict]) -> list[Path]:
    """Draw the full configured radius scan at a readable single-column size."""
    apply_lps_style()
    fig, ax = plt.subplots(figsize=(3.40, 2.8), layout="constrained")
    labels = [r"$f_{2\pi/3}$", r"$f_m$", r"$f_{\pi/2}$"]
    for marker, label, symbol, color in zip(config["markers"], labels, ("o", "s", "^"), ("C0", "C1", "C2")):
        points = sorted((row for row in rows if row["marker"] == marker["name"]), key=lambda row: row["r_c_mm"])
        ax.plot([row["r_c_mm"] for row in points], [row["unwrapped_phase_deg"] for row in points],
                label=label, color=color, lw=2, marker=symbol, markevery=20,
                markersize=5, markerfacecolor="white", markeredgewidth=1.2)
    nominal = config["radius_scan"]["nominal_radius"]
    ax.axvline(nominal, color="0.4", lw=0.9, ls="--", zorder=0)
    grid = config["radius_scan"]["expected_grid"]
    padding = 2 * grid["step"]
    ax.set(xlabel="$r_c$ [mm]", ylabel="$S_{11}$ phase [deg]",
           xlim=(grid["minimum"] - padding, grid["maximum"] + padding))
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(color="0.90", lw=0.6)
    ax.legend(loc="lower left", handlelength=1.6, handletextpad=0.4)
    finish_axes(ax)
    outputs = []
    for suffix in ("pdf", "png"):
        path = output_root / "fig" / f"fig05-iris-radius-response.{suffix}"
        metadata = {"Creator": "build_paper_cst_bundle.py"}
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        save_paper_figure(fig, path, metadata=metadata)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_cst_20261003.json")
    parser.add_argument("--output-root", type=Path, default=ROOT.parent / "TDC-AcademicPaper")
    args = parser.parse_args(argv)
    config_path, output_root = args.config.resolve(), args.output_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Validate all sources and pairs before writing any publication product.
    metrics, geometry, audit = make_tables(ROOT, config)
    radius_rows, radius_audit = make_radius_table(ROOT, config)
    outputs = [output_root / "tables" / "paper-short-state-metrics.csv",
               output_root / "tables" / "paper-geometry-summary.csv",
               output_root / "tables" / "paper-radius-response.csv"]
    write_csv(outputs[0], metrics)
    write_csv(outputs[1], geometry)
    write_csv(outputs[2], radius_rows)
    outputs.extend(draw_figures(output_root, config, metrics))
    outputs.extend(draw_radius_figure(output_root, config, radius_rows))
    source_files = [Path(__file__).resolve(), config_path,
                    ROOT / "deflector_tuning/data_loading/readers/touchstone_reader.py",
                    ROOT / "deflector_tuning/data_loading/loaders/sim_loader.py",
                    ROOT / "deflector_tuning/visualization/lps_paper_style.py",
                    ROOT / "config/lps_publication_style.json"]
    manifest = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "data_kind": "simulation", "config": config,
                "generator": {"argv": [sys.executable, *sys.argv],
                              "resolved_config": str(config_path), "resolved_output_root": str(output_root),
                              "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                              "source_hashes": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
                                                for path in source_files]},
                "validation": {"source_count": sum(case["selected_count"] for case in audit["cases"]),
                               "source_count_scope": "Short-state sources; radius sources counted separately",
                               "all_primary_source_count": sum(case["selected_count"] for case in audit["cases"]) + radius_audit["source_count"],
                               "metrics_row_count": len(metrics), "geometry_row_count": len(geometry),
                               "radius_source_count": radius_audit["source_count"],
                               "radius_row_count": len(radius_rows),
                               "all_pair_metadata_consistent": True,
                               "all_selected_sources_are_cst_simulation": True,
                               "sample_selection": "Nearest exported frequency; no interpolation; first row wins ties",
                               "no_impedance_conversion_or_renormalization": True},
                **audit,
                "radius_scan": radius_audit,
                "outputs": [{"path": path.relative_to(output_root).as_posix(), "sha256": sha256(path)}
                            for path in outputs]}
    manifest_path = output_root / "docs" / "evidence" / "cst-bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in metrics:
        print(f"{row['case']:4s} {row['pair']:4s} {row['marker']:7s}: "
              f"delta={row['delta_deg']:.8f}, residual={row['signed_target_residual_deg']:.8f}, "
              f"sample={row['first_actual_frequency_GHz']:.9f} GHz")
    for marker, response in radius_audit["phase_responses"].items():
        print(f"radius {marker}: nominal centered slope="
              f"{response['nominal_central_slope_deg_per_mm']:.8f} deg/mm")
    print(manifest_path)
    return manifest_path


if __name__ == "__main__":
    main()
