"""Draw reference/first-tuning polar phase overlays from audited paper points.

This follows the existing polar_overlays phase-only display: dotted/open
reference states, solid/filled tuning states, and signed shortest arcs.
Display radii distinguish frequencies and never represent S11 magnitude.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.visualization.lps_paper_style import (
    PRESET_PATH, apply_lps_style, save_paper_figure,
)
from deflector_tuning.visualization.polar_drawing import _wrap180


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT.parent / "TDC-AcademicPaper")
    parser.add_argument("--config", type=Path, default=ROOT / "config/paper_cst_20261003.json")
    args = parser.parse_args()
    paper = args.paper_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = paper / "tables/paper-short-state-metrics.csv"
    before = digest(source)
    with source.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    apply_lps_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 6.4), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=.06, right=.96, bottom=.04, top=.79, wspace=.25, hspace=.60)
    colors = ["C0", "C1", "C2"]
    symbols = ["o", "s", "^"]
    labels = [r"$f_{2\pi/3}$", r"$f_m$", r"$f_{\pi/2}$"]
    points, source_hashes, loaded = [], {}, {}
    for row_index, case in enumerate(config["cases"]):
        for col_index, pair in enumerate(config["pairs"]):
            ax = axes[row_index, col_index]
            selected = [next(r for r in rows if r["case"] == case["id"]
                             and r["pair"] == pair["name"] and r["marker"] == marker["name"])
                        for marker in config["markers"]]
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_ylim(0, 1.08)
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]),
                          [r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"])
            ax.set_yticks([.5, .7, .9], [])
            ax.grid(color=".78", linewidth=.6)
            ax.spines["polar"].set_linewidth(1)
            ax.tick_params(labelsize=8, pad=1)
            case_label = ("Extended", "Reduced")[row_index]
            letter = chr(97 + 2 * row_index + col_index)
            ax.set_title(f"({letter}) {case_label}: {pair['label']}", fontsize=10.5, pad=16)
            for point, radius, color, symbol, label in zip(selected, [.5, .7, .9], colors, symbols, labels):
                phases = []
                for state in ("first", "second"):
                    path = ROOT / point[f"{state}_source"]
                    if path not in loaded:
                        source_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
                        loaded[path] = read_touchstone(path)
                    assert source_hashes[path.relative_to(ROOT).as_posix()] == point[f"{state}_sha256"]
                    trace = loaded[path]
                    index = int(point[f"{state}_sample_index_zero_based"])
                    value = complex(float(point[f"{state}_real"]), float(point[f"{state}_imag"]))
                    assert trace.values[index][0] == value
                    phase = float(np.angle(value, deg=True))
                    np.testing.assert_allclose(phase, float(point[f"{state}_phase_deg"]), atol=1e-12)
                    phases.append(phase)
                    angle = np.deg2rad(phase)
                    ax.plot([angle, angle], [0, 1], color=color, alpha=.50,
                            ls=":" if state == "first" else "-", lw=1.2)
                    ax.plot(angle, radius, marker=symbol, ms=5.5, ls="none",
                            color=color, mfc="white" if state == "first" else color,
                            mew=1.2, zorder=4)
                    points.append({"case":case["id"], "pair":pair["name"], "marker":point["marker"],
                                   "state":state, "NumDepth":float(point[f"{state}_NumDepth"]),
                                   "real":value.real, "imag":value.imag, "phase_deg":phase,
                                   "display_radius":radius})
                delta = float(_wrap180(phases[1] - phases[0]))
                np.testing.assert_allclose(delta, _wrap180(float(point["delta_deg"])), atol=1e-10)
                theta = np.deg2rad(np.linspace(phases[0], phases[0]+delta, 120))
                ax.plot(theta, np.full_like(theta, radius), color=color, lw=2, alpha=.85)
    frequency_handles = [Line2D([], [], color=c, marker=s, lw=2, ms=5) for c,s in zip(colors,symbols)]
    fig.legend(frequency_handles, labels, loc="upper center", bbox_to_anchor=(.51,.995),
               ncol=3, columnspacing=2, handlelength=2)
    role_handles = [Line2D([],[],color=".25",marker="o",mfc="white",ls=":",lw=1.5),
                    Line2D([],[],color=".25",marker="o",mfc=".25",ls="-",lw=1.5)]
    fig.legend(role_handles, ["Reference plane: $C_0$ / $I_0$", "First tuning plane: $C_1$ / $I_1$"],
               loc="upper center", bbox_to_anchor=(.51,.937), ncol=2, columnspacing=1.1)
    outputs = []
    for suffix in ("pdf", "png"):
        path = paper / "fig" / f"fig-cst-polar-planes.{suffix}"
        save_paper_figure(fig, path)
        outputs.append(path)
    plt.close(fig)
    assert len(points) == 24
    assert digest(source) == before
    record = {"input_metric_table_sha256":before, "source_sha256":source_hashes,
              "point_count":len(points), "points":points,
              "angle_definition":"raw exported S11 phase; no common rotation or de-embedding",
              "arc_definition":"signed shortest phase change in [-180,180); modulo 360 equals paper delta",
              "radius_definition":"display only: 0.5/0.7/0.9 for the three frequencies; not magnitude",
              "reference_plane_role":"first plunger position of each pair, not the external port reference plane",
              "code_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in
                             [Path(__file__), PRESET_PATH, ROOT/"deflector_tuning/visualization/lps_paper_style.py",
                              ROOT/"deflector_tuning/visualization/polar_drawing.py"]},
              "outputs":[{"path":p.relative_to(paper).as_posix(),"sha256":digest(p)} for p in outputs]}
    (paper/"docs/evidence/polar-comparison-manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print("PASS: 24 polar endpoints match 8 raw CST files and all 12 signed pair differences")


if __name__ == "__main__":
    main()
