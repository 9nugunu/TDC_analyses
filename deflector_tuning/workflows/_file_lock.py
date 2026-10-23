"""Process-safe advisory lock for local read-modify-replace transactions."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time
from typing import Iterator


@contextmanager
def exclusive_file_lock(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Hold an OS-released lock on a stable sibling file until context exit.

    The lock file is intentionally retained: deleting it would let new callers
    lock a different inode while existing callers are still waiting. All
    cooperating writers must use the same path. A process crash releases the
    OS lock; an abandoned lock file therefore never blocks future writers.
    """
    if os.name == "nt":
        import msvcrt
    else:
        import fcntl

    deadline = time.monotonic() + timeout_seconds
    with path.open("a+b") as handle:
        while True:
            try:
                if os.name == "nt":
                    handle.seek(0)
                    # Windows permits locking a byte beyond the end of a file.
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for file lock {path}") from error
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
