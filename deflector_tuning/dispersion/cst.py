"""Process CST dispersion text exports into reusable CSV tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODE_HEADER_RE = re.compile(r'"Mode\s+(?P<mode>\d+)\s+\[Real\s*/\s*GHz\]"')
KEY_PHASES_DEG: tuple[float, ...] = (0.0, 90.0, 120.0, 180.0)


@dataclass(frozen=True)
class DispersionOutputs:
    """CSV paths produced from one CST dispersion export."""

    long_csv: Path
    wide_csv: Path
    summary_csv: Path


def load_cst_dispersion_txt(path: str | Path) -> pd.DataFrame:
    """Load a CST dispersion text export into a long table.

    CST exports used here are one or more repeated sections:
    ``"phase"`` followed by ``"Mode N [Real / GHz]"`` and numeric rows.
    """

    source_path = Path(path)
    rows: list[dict[str, object]] = []
    mode_index: int | None = None
    with source_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = MODE_HEADER_RE.search(line)
            if match:
                mode_index = int(match.group("mode"))
                continue
            if line.startswith("#"):
                continue
            if mode_index is None:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                phase_deg = float(parts[0])
                freq_ghz = float(parts[1])
            except ValueError:
                continue
            rows.append(
                {
                    "source_file": source_path.name,
                    "mode_index": mode_index,
                    "phase_deg": phase_deg,
                    "freq_GHz": freq_ghz,
                }
            )

    if not rows:
        raise ValueError(f"No CST dispersion rows found in {source_path}")
    table = pd.DataFrame(rows)
    return table.sort_values(["mode_index", "phase_deg"], kind="mergesort").reset_index(drop=True)


def to_dispersion_wide_table(long_table: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long dispersion table to one row per phase."""

    _require_columns(long_table, ("mode_index", "phase_deg", "freq_GHz"), context="long_table")
    wide = long_table.pivot_table(index="phase_deg", columns="mode_index", values="freq_GHz", aggfunc="first")
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    wide.columns = [f"mode_{int(column):02d}_GHz" for column in wide.columns]
    return wide.reset_index()


def summarize_dispersion_modes(long_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize each mode and extract key marker frequencies."""

    _require_columns(long_table, ("mode_index", "phase_deg", "freq_GHz"), context="long_table")
    summary_rows: list[dict[str, object]] = []
    for mode_index, group in long_table.groupby("mode_index", sort=True):
        group = group.sort_values("phase_deg")
        phases = group["phase_deg"].to_numpy(dtype=float)
        freqs = group["freq_GHz"].to_numpy(dtype=float)
        key_freqs = {phase: _frequency_at_phase(phases, freqs, phase) for phase in KEY_PHASES_DEG}
        fit = _fit_single_chain_cosine(phases, freqs)
        summary_rows.append(
            {
                "mode_index": int(mode_index),
                "npoints": int(len(group)),
                "phase_min_deg": float(np.min(phases)),
                "phase_max_deg": float(np.max(phases)),
                "freq_min_GHz": float(np.min(freqs)),
                "freq_max_GHz": float(np.max(freqs)),
                "freq_0_GHz": key_freqs[0.0],
                "freq_90_GHz": key_freqs[90.0],
                "freq_120_GHz": key_freqs[120.0],
                "freq_180_GHz": key_freqs[180.0],
                "delta_120_minus_90_MHz": (key_freqs[120.0] - key_freqs[90.0]) * 1000.0,
                "monotonicity": _monotonicity(freqs),
                **fit,
            }
        )
    return pd.DataFrame(summary_rows)


def process_cst_dispersion_txt(path: str | Path, output_dir: str | Path | None = None) -> DispersionOutputs:
    """Write long, wide, and summary CSVs for one CST dispersion text export."""

    source_path = Path(path)
    folder = Path(output_dir) if output_dir is not None else source_path.parent / "processed"
    folder.mkdir(parents=True, exist_ok=True)

    long_table = load_cst_dispersion_txt(source_path)
    wide_table = to_dispersion_wide_table(long_table)
    summary_table = summarize_dispersion_modes(long_table)

    stem = source_path.stem
    long_csv = folder / f"{stem}_long.csv"
    wide_csv = folder / f"{stem}_wide.csv"
    summary_csv = folder / f"{stem}_summary.csv"
    long_table.to_csv(long_csv, index=False)
    wide_table.to_csv(wide_csv, index=False)
    summary_table.to_csv(summary_csv, index=False)
    return DispersionOutputs(long_csv=long_csv, wide_csv=wide_csv, summary_csv=summary_csv)


def _frequency_at_phase(phases: np.ndarray, freqs: np.ndarray, phase_deg: float) -> float:
    if phase_deg < float(np.min(phases)) or phase_deg > float(np.max(phases)):
        return float("nan")
    order = np.argsort(phases)
    return float(np.interp(phase_deg, phases[order], freqs[order]))


def _monotonicity(freqs: np.ndarray) -> str:
    diffs = np.diff(freqs)
    if np.all(diffs >= 0):
        return "increasing"
    if np.all(diffs <= 0):
        return "decreasing"
    return "mixed"


def _fit_single_chain_cosine(phases: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(phases) & np.isfinite(freqs) & (freqs > 0)
    phases = phases[finite]
    freqs = freqs[finite]
    if len(freqs) < 2:
        return _empty_fit()

    x = np.cos(np.deg2rad(phases))
    y = 1.0 / np.square(freqs)
    design = np.column_stack([np.ones_like(x), x])
    try:
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _empty_fit()
    if intercept <= 0:
        return _empty_fit()

    f0 = float(1.0 / np.sqrt(intercept))
    k = float(slope / intercept)
    denominator = 1.0 + k * x
    if np.any(denominator <= 0):
        return _empty_fit()

    predicted = f0 / np.sqrt(denominator)
    residual_mhz = (freqs - predicted) * 1000.0
    total = float(np.sum(np.square((freqs - np.mean(freqs)) * 1000.0)))
    residual = float(np.sum(np.square(residual_mhz)))
    r2 = float("nan") if total == 0.0 else 1.0 - residual / total
    return {
        "single_chain_f0_GHz": f0,
        "single_chain_k": k,
        "single_chain_rmse_MHz": float(np.sqrt(np.mean(np.square(residual_mhz)))),
        "single_chain_max_abs_MHz": float(np.max(np.abs(residual_mhz))),
        "single_chain_r2": r2,
    }


def _empty_fit() -> dict[str, float]:
    return {
        "single_chain_f0_GHz": float("nan"),
        "single_chain_k": float("nan"),
        "single_chain_rmse_MHz": float("nan"),
        "single_chain_max_abs_MHz": float("nan"),
        "single_chain_r2": float("nan"),
    }


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], *, context: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")
