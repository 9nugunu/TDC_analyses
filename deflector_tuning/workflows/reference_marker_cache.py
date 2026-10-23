"""Content signatures for safely reusing reference marker tables."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import pandas as pd
from pathlib import Path

from deflector_tuning.project_defaults import ProjectDefaults


def marker_reference_signature(*, source: Path, data_root: Path,
                               project_defaults: ProjectDefaults,
                               marker_role: str, correction: object,
                               dispersion_path: str | Path | None = None) -> str:
    """Fingerprint source contents and effective scientific settings.

    Legacy tables without a signature are regenerated once. Hashing is read-only
    and does not require repeating the numerical analysis of unchanged inputs.
    """
    from deflector_tuning.runner import resolve_input_paths

    source, dispersion = resolve_input_paths(
        sparameter_path=source, dispersion_path=dispersion_path, data_root=data_root,
        default_dispersion_subpath=project_defaults.default_dispersion_subpath,
    )
    payload = {
        "version": 1, "role": marker_role, "defaults": asdict(project_defaults),
        "correction": asdict(correction) if correction is not None else None,
        "sources": _content_hashes(source), "dispersion": _content_hashes(dispersion),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _content_hashes(path: Path) -> list[tuple[str, str]]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        return [(str(path.resolve()), "missing")]
    return [(str(item.resolve()), sha256(item.read_bytes()).hexdigest()) for item in files]


def record_marker_signature(marker_path: Path, signature: str) -> None:
    """Bind input signature to the exact table bytes, detecting replacement too."""
    payload = {"inputs": signature, "table": _table_digest(marker_path)}
    marker_path.with_suffix(".signature.json").write_text(json.dumps(payload), encoding="utf-8")


def marker_signature_matches(marker_path: Path, signature: str) -> bool:
    try:
        payload = json.loads(marker_path.with_suffix(".signature.json").read_text(encoding="utf-8"))
        return payload == {"inputs": signature, "table": _table_digest(marker_path)}
    except (OSError, ValueError):
        return False


def marker_table_constants(marker_path: Path) -> dict:
    manifest = marker_path.parent.parent / "manifest.json"
    if not manifest.is_file():
        return {}
    return json.loads(manifest.read_text(encoding="utf-8")).get("table_constants", {}).get("marker_pts", {})


def _table_digest(marker_path: Path) -> str:
    digest = sha256(marker_path.read_bytes())
    digest.update(json.dumps(marker_table_constants(marker_path), sort_keys=True).encode())
    return digest.hexdigest()


def read_marker_table(marker_path: Path) -> pd.DataFrame:
    """Restore projected constant metadata such as the simulation width."""
    table = pd.read_csv(marker_path)
    for name, value in marker_table_constants(marker_path).items():
        if name not in table:
            table[name] = value
    return table
