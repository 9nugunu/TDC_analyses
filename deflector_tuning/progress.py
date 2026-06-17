"""Optional progress-bar helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress_iter(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Iterator[T]:
    """Return ``iterable`` wrapped in tqdm when it is available."""

    if os.environ.get("DEFLECTOR_TUNING_PROGRESS", "").lower() in {"0", "false", "no", "off"}:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterable
        return

    yield from tqdm(iterable, desc=desc, total=total, unit="item")
