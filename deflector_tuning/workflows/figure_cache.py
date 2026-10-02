"""Content identities for reusing S11 figures from an earlier analysis run."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from deflector_tuning.visualization.plot_config import PlotConfig


def s11_figure_signature(
    sparameter_table: pd.DataFrame,
    marker_points: pd.DataFrame,
    config: PlotConfig,
) -> str:
    """Hash loaded values, marker conditions, style, and the rendering code.

    Frames are hashed in bounded chunks, using the data already loaded for
    analysis. No raw file is reopened. Package source identity conservatively
    invalidates images after code edits as well as numerical input changes.
    """
    digest = hashlib.sha256()
    settings = {
        "schema": 1,
        "plot_config": asdict(config),
        "source": _renderer_source_signature(),
        "runtime": [matplotlib.__version__, np.__version__, pd.__version__],
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for frame in (sparameter_table, marker_points):
        schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
        digest.update(json.dumps([schema, len(frame)]).encode("utf-8"))
        for start in range(0, len(frame), 50_000):
            chunk = frame.iloc[start : start + 50_000]
            digest.update(pd.util.hash_pandas_object(chunk, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _renderer_source_signature() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
