"""Cache identity follows the numerical inputs and effective rendering settings."""
from dataclasses import replace
from pathlib import Path
import json

import pandas as pd
import pytest

from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.workflows.figure_cache import s11_figure_signature
from deflector_tuning.workflows.manifest import cached_manifest_figure_group


def _inputs():
    trace = pd.DataFrame({"source_file": ["run_1.s1p"] * 2,
                          "freq_ghz": [2.85, 2.86], "s_db": [-1.0, -2.0]})
    markers = pd.DataFrame({"source_file": ["run_1.s1p"], "marker_name": ["f_2pi3"],
                            "freq_ghz": [2.85], "s_db": [-1.0]})
    return trace, markers, PlotConfig()


@pytest.mark.parametrize("changed", ["trace", "marker", "style", "source_label"])
def test_s11_signature_changes_with_effective_inputs(changed):
    trace, markers, config = _inputs()
    original = s11_figure_signature(trace, markers, config)
    assert original == s11_figure_signature(trace.copy(), markers.copy(), config)
    if changed == "trace":
        trace.loc[1, "s_db"] = -4.0
    elif changed == "marker":
        markers.loc[0, "freq_ghz"] = 2.86
    elif changed == "style":
        config = replace(config, dpi=config.dpi + 1)
    else:
        trace["source_file"] = "run_2.s1p"
    assert s11_figure_signature(trace, markers, config) != original


def test_s11_signature_changes_when_renderer_changes(monkeypatch):
    import deflector_tuning.workflows.figure_cache as cache
    trace, markers, config = _inputs()
    monkeypatch.setattr(cache, "_renderer_source_signature", lambda: "old-renderer")
    original = s11_figure_signature(trace, markers, config)
    monkeypatch.setattr(cache, "_renderer_source_signature", lambda: "new-renderer")
    assert s11_figure_signature(trace, markers, config) != original


@pytest.mark.parametrize("stored", [None, "old-key", "current-key"])
def test_cached_figure_requires_matching_signature(tmp_path: Path, stored):
    figure = tmp_path / "s11.png"
    figure.write_bytes(b"existing image")
    payload = {"outputs": {"figures": {"s11": {"overview": str(figure)}}}}
    if stored is not None:
        payload["figure_cache"] = {"s11": stored}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    actual = cached_manifest_figure_group(manifest, "s11", expected_signature="current-key")
    assert actual == ({"overview": figure} if stored == "current-key" else {})
    figure.unlink()
    assert not cached_manifest_figure_group(manifest, "s11", expected_signature="current-key")
