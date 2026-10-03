"""Build paper field sections, sampled dispersion, and combined CST plots.

All inputs are existing simulation exports. Existing paper tables and figures
are preserved. Field panels show shape with independent per-panel maxima.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deflector_tuning.analysis.field_sections import load_field_section, normalized_field_magnitude
from deflector_tuning.data_loading.field3d import find_field3d_pairs, EPSILON_0_F_PER_M, MU_0_H_PER_M
from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.dispersion.cst import load_cst_dispersion_txt
from deflector_tuning.visualization.lps_paper_style import (
    PRESET_PATH, apply_lps_style, field_colormap, finish_axes, save_paper_figure,
)

LOGGER = logging.getLogger(__name__)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, paper, stem):
    paths = []
    for extension in ("pdf", "png"):
        path = paper / "fig" / f"{stem}.{extension}"
        save_paper_figure(fig, path)
        paths.append(path)
    plt.close(fig)
    return paths


def field_figure(root, paper, config):
    dataset = root / config["field_dataset"]
    pairs = {pair.case_id: pair for pair in find_field3d_pairs(dataset)}
    with (root / config["cached_energy_table"]).open(encoding="utf-8") as stream:
        cached = {row["case_id"]: row for row in csv.DictReader(stream)}
    fig, axes = plt.subplots(2, len(config["field_cases"]), figsize=(7.08, 3.35),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.15, top=0.89,
                        wspace=0.12, hspace=0.18)
    cmap = field_colormap()
    records, sections, arrays = [], [], {}
    reference_grid = None
    for col, case in enumerate(config["field_cases"]):
        pair = pairs[case["id"]]
        loaded = []
        for kind, path, factor in (("e", pair.e_path, EPSILON_0_F_PER_M),
                                   ("h", pair.h_path, MU_0_H_PER_M)):
            LOGGER.info("Reading %s %s", case["id"], kind.upper())
            before = digest(path)
            section = load_field_section(path, fixed_axis=config["fixed_axis"],
                                         coordinate_mm=config["requested_coordinate_mm"])
            assert before == digest(path), "Raw file changed during reading"
            if reference_grid is None:
                reference_grid = section
            assert section.actual_coordinate_mm == reference_grid.actual_coordinate_mm
            np.testing.assert_array_equal(section.transverse_mm, reference_grid.transverse_mm)
            np.testing.assert_array_equal(section.z_mm, reference_grid.z_mm)
            assert section.full_grid_shape == reference_grid.full_grid_shape
            energy = float(0.25 * factor * section.full_component_abs2_sums.sum()
                           * np.prod(section.full_spacing_mm) * 1e-9)
            np.testing.assert_allclose(energy, float(cached[case["id"]][f"{kind}_region_j"]), rtol=1e-10)
            norm, peak = normalized_field_magnitude(section.values)
            arrays[f"{case['id']}_{kind}_complex"] = section.values
            arrays[f"{case['id']}_{kind}_normalized"] = norm
            records.append({"case_id": case["id"], "field": kind,
                            "source": path.relative_to(root).as_posix(), "sha256": before,
                            "row_count": section.row_count, "grid_shape": section.full_grid_shape,
                            "bounds_mm": section.full_bounds_mm,
                            "actual_coordinate_mm": section.actual_coordinate_mm,
                            "plane_peak_native_units": peak, "export_region_energy_j": energy,
                            "cached_energy_reproduced": True})
            loaded.append((section, norm))
        combined = sum(np.linalg.norm(section.values, axis=-1) ** 2 for section, _ in loaded)
        zeros = combined == 0
        iy = int(np.argmin(np.abs(loaded[0][0].transverse_mm)))
        center_zero = zeros[iy]
        candidates = [i for i in range(len(center_zero)-1)
                      if center_zero[i:].all() and len(center_zero)-i >= 2]
        tip = float(loaded[0][0].z_mm[candidates[0]]) if candidates else None
        sections.append({"case_id": case["id"], "label": case["label"],
                         "depth_label": case["depth"], "axial_zero_region_start_mm": tip,
                         "zero_samples_in_cut": int(zeros.sum()),
                         "normalization": "own plane maximum, separately for E and H"})
        for row, (section, norm) in enumerate(loaded):
            ax = axes[row, col]
            mesh = ax.pcolormesh(section.z_mm, section.transverse_mm,
                                 np.ma.masked_where(zeros, norm), cmap=cmap,
                                 vmin=0, vmax=1, shading="nearest", rasterized=True)
            if tip is not None:
                ax.axvline(tip, color="white", lw=0.8, ls="--")
            ax.set_aspect("equal")
            ax.set_xticks([0, 50, 100])
            ax.set_yticks([-50, 0, 50])
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(case["label"], fontsize=8.5, pad=6)
            if col == 0:
                ax.set_ylabel((r"$|\mathbf{E}|$" if row == 0 else r"$|\mathbf{H}|$") + "\n$y$ [mm]")
            if row == 1:
                ax.set_xlabel("$z$ [mm]")
            finish_axes(ax, scale=0.8, grid=False)
    cax = fig.add_axes([0.90, 0.22, 0.014, 0.58])
    colorbar = fig.colorbar(mesh, cax=cax, ticks=[0, 0.5, 1])
    cax.tick_params(labelsize=8)
    colorbar.set_label("Normalized magnitude", fontsize=9.2, labelpad=7)
    arrays["transverse_mm"] = reference_grid.transverse_mm
    arrays["z_mm"] = reference_grid.z_mm
    array_path = paper / "tables/paper-field-sections.npz"
    np.savez_compressed(array_path, **arrays)
    paths = save(fig, paper, "fig-fields-short-states") + [array_path]
    return paths, {"sources": records, "sections": sections,
                   "excitation_normalization_verified": False,
                   "same_geometry_as_june_scattering_verified": False,
                   "field_monitor_frequency": "2.857 GHz in filename; full precision unavailable",
                   "reference_case": "NoPlunger label; navigator run 5 has NumDepth=14",
                   "navigator_sha256": digest(dataset / "result_navigator.csv")}


def dispersion_figure(root, paper, config, markers):
    source = root / config["dispersion_source"]
    table = load_cst_dispersion_txt(source)
    data = table[table.mode_index == config["mode_index"]].copy()
    assert not data.phase_deg.duplicated().any()
    assert len(data) == 181
    values = dict(zip(data.phase_deg, data.freq_GHz))
    np.testing.assert_allclose([values[120], (values[90]+values[120])/2, values[90]],
                               [m["requested_GHz"] for m in markers], rtol=0, atol=1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.85), layout="constrained")
    for ax in axes:
        ax.plot(data.phase_deg, data.freq_GHz, color="C0", lw=2)
        ax.plot([90, 120], [values[90], values[120]], "o", color="C1", ms=5)
        ax.set_xlabel("Cell phase advance [deg]")
    axes[0].set(xlim=(0,180), xticks=[0,60,90,120,180], ylabel="Frequency [GHz]")
    axes[0].set_title("(a) CST Mode 1", fontsize=9)
    axes[1].set(xlim=(80,130), ylim=(2.851,2.892), xticks=[90,105,120])
    axes[1].set_title("(b) Reference frequencies")
    for frequency, label in ((values[90], r"$f_{\pi/2}$"),
                              ((values[90]+values[120])/2,r"$f_m$"),
                              (values[120],r"$f_{2\pi/3}$")):
        axes[1].axhline(frequency,color="0.5",ls="--",lw=0.7)
        axes[1].text(129,frequency+0.0007,label,ha="right",va="bottom",fontsize=10)
    for ax in axes:
        finish_axes(ax)
    paths = save(fig,paper,"fig-cst-dispersion-markers")
    out = paper / "tables/paper-dispersion-mode1.csv"
    data.to_csv(out,index=False)
    return paths+[out], {"source":config["dispersion_source"],"sha256":digest(source),
                         "mode_index":config["mode_index"],"point_count":len(data),
                         "geometry_identity_with_driven_cases_verified":False}


def reflection_figure(root,paper,config):
    source=paper/"tables/paper-short-state-metrics.csv"
    before=digest(source)
    with source.open(encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    marks=np.array([m["requested_GHz"] for m in config["markers"]])
    span=marks[-1]-marks[0]
    limits=(marks[0]-0.15*span,marks[-1]+0.15*span)
    fig,axes=plt.subplots(2,2,figsize=(7.08,4.8),sharey="row")
    fig.subplots_adjust(left=.12,right=.985,bottom=.08,top=.82,wspace=.14,hspace=.44)
    traces,inputs=[],{}
    for col,case in enumerate(config["cases"]):
        axes[0,col].set_title(("(a) Extended" if col==0 else "(b) Reduced"),fontsize=9,pad=6)
        for pair_index,pair in enumerate(config["pairs"]):
            selected=[next(r for r in rows if r["case"]==case["id"] and r["pair"]==pair["name"] and r["marker"]==m["name"]) for m in config["markers"]]
            for state_index,state in enumerate(("first","second")):
                row=selected[0]
                path=root/row[f"{state}_source"]
                assert digest(path)==row[f"{state}_sha256"]
                inputs[path.relative_to(root).as_posix()]=digest(path)
                trace=read_touchstone(path)
                scales={"Hz":1e-9,"kHz":1e-6,"MHz":1e-3,"GHz":1.0}
                freq=np.asarray(trace.frequency)*scales[trace.header.frequency_unit]
                values=np.asarray(trace.values)[:,0]
                for point in selected:
                    idx=int(point[f"{state}_sample_index_zero_based"])
                    assert values[idx]==complex(float(point[f"{state}_real"]),float(point[f"{state}_imag"]))
                keep=(freq>=limits[0])&(freq<=limits[1])
                phase=np.angle(values[keep],deg=True)
                display=phase.copy()
                display[np.r_[False,np.abs(np.diff(display))>180]]=np.nan
                label=row[f"{state}_label"]
                axes[0,col].plot(freq[keep],display,color=f"C{pair_index}",
                                 ls="--" if state_index==0 else "-",lw=2,label=label)
                for f,v in zip(freq[keep],values[keep]):
                    traces.append({"case":case["id"],"state":label,"frequency_GHz":f,"real":v.real,"imag":v.imag})
            offset=(-.06,.06)[pair_index]
            axes[1,col].plot(np.arange(3)+offset,[float(r["signed_target_residual_deg"]) for r in selected],
                             ls="none",marker=("o","s")[pair_index],color=f"C{pair_index}",ms=5,
                             markerfacecolor="white", markeredgewidth=1.4, label=pair["label"])
        for m in marks:
            axes[0,col].axvline(m,color="0.65",ls=":",lw=.7,zorder=0)
        axes[0,col].set(xlim=limits,ylim=(-195,195),yticks=[-180,-90,0,90,180],xlabel="Frequency [GHz]")
        axes[1,col].set(xticks=[0,1,2],xticklabels=[r"$f_{2\pi/3}$",r"$f_m$",r"$f_{\pi/2}$"],
                        ylim=(-180,180),yticks=[-150,0,150],xlim=(-.3,2.3))
        axes[1,col].axhline(0,color=".4",ls="--",lw=.8)
    axes[0,0].set_ylabel("Reflection phase [deg]")
    axes[1,0].set_ylabel("Reference residual [deg]")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,bbox_to_anchor=(.55,.99))
    axes[1,1].legend(loc="lower left")
    for ax in axes.flat:
        finish_axes(ax)
    paths=save(fig,paper,"fig-cst-phase-comparison")
    out=paper/"tables/paper-reflection-curves.csv"
    write_rows(out,traces)
    assert before==digest(source)
    return paths+[out],{"metric_table_sha256":before,"source_sha256":inputs,
                        "curve_rows":len(traces),"frequency_limits_GHz":limits,
                        "display_rule":"Wrapped phase; lines break at >180 degree jumps; no smoothing"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root",type=Path,default=ROOT)
    parser.add_argument("--paper-root",type=Path,default=ROOT.parent/"TDC-AcademicPaper")
    parser.add_argument("--config",type=Path,default=ROOT/"config/paper_fields_20261003.json")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    root,paper=args.analysis_root.resolve(),args.paper_root.resolve()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    scattering=json.loads((root/config["scattering_config"]).read_text(encoding="utf-8"))
    protected={path:digest(path) for path in (paper/"tables").glob("*.csv")}
    for folder in ("fig","tables","docs/evidence"):(paper/folder).mkdir(parents=True,exist_ok=True)
    apply_lps_style()
    outputs,fields=field_figure(root,paper,config)
    paths,dispersion=dispersion_figure(root,paper,config,scattering["markers"]); outputs+=paths
    paths,reflection=reflection_figure(root,paper,scattering); outputs+=paths
    unchanged={path.name:digest(path)==value for path,value in protected.items()
               if path not in outputs}
    assert all(unchanged.values())
    record={"generated_utc":datetime.now(timezone.utc).isoformat(),"data_kind":"simulation",
            "config":config,"field_comparison":fields,"dispersion":dispersion,"reflection":reflection,
            "previous_numerical_tables_unchanged":unchanged,
            "code_sha256":{str(Path(__file__).relative_to(root)):digest(__file__),
                            "deflector_tuning/analysis/field_sections.py":digest(root/"deflector_tuning/analysis/field_sections.py"),
                            "deflector_tuning/visualization/lps_paper_style.py":digest(root/"deflector_tuning/visualization/lps_paper_style.py"),
                            PRESET_PATH.relative_to(ROOT).as_posix():digest(PRESET_PATH)},
            "outputs":[{"path":p.relative_to(paper).as_posix(),"sha256":digest(p)} for p in outputs]}
    (paper/"docs/evidence/field-comparison-manifest.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    LOGGER.info("PASS: five E/H cases, dispersion and combined reflection plots; prior numeric tables unchanged")


if __name__=="__main__": main()
