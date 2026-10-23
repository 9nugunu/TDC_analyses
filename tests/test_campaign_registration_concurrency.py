"""Cross-process and failure regressions for automatic campaign registration."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from deflector_tuning.workflows.tuning_campaign import _persist_automatic_measurement
from deflector_tuning.workflows._file_lock import exclusive_file_lock


def _match(path: Path, *, kind: str = "aux", state_id: str = "s001"):
    state = SimpleNamespace(
        previous_state_id="s000", measurement_status="done",
        quality_flag="auto_pending", torque_nm=None,
    )
    return SimpleNamespace(
        config_path=path, state_id=state_id, measurement_kind=kind,
        campaign=SimpleNamespace(states={state_id: state}),
    )


def _register_process(path, name, first, started, read, release, first_done):
    # Pause after reading the snapshot. With a lock, the second reader waits
    # outside this hook. Without one it retains a stale snapshot until first exits.
    original_load = yaml.safe_load

    def synchronized_load(handle):
        result = original_load(handle)
        handle.close()  # Windows must permit replacement while snapshot waits.
        read.set()
        event = release if first else first_done
        if not event.wait(20):
            raise RuntimeError("registration synchronization timed out")
        return result

    yaml.safe_load = synchronized_load
    started.set()
    try:
        _persist_automatic_measurement(_match(Path(path)), name)
    finally:
        if first:
            first_done.set()


def test_parallel_processes_preserve_both_registration_records(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text("states: {}\naux: {}\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    release, first_done = context.Event(), context.Event()
    started = [context.Event(), context.Event()]
    read = [context.Event(), context.Event()]
    workers = [
        context.Process(target=_register_process, args=(
            str(path), f"dataset_{index}", index == 0, started[index],
            read[index], release, first_done,
        )) for index in range(2)
    ]
    try:
        workers[0].start()
        assert read[0].wait(20)
        workers[1].start()
        assert started[1].wait(20)
        # Old implementation reaches this event and overwrites the first
        # record. The corrected implementation blocks until first releases.
        second_read_before_release = read[1].wait(2)
        release.set()
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(raw["aux"]) == {"dataset_0", "dataset_1"}
        assert not second_read_before_release
    finally:
        release.set()
        first_done.set()
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(5)


def _hold_lock_process(path, acquired):
    with exclusive_file_lock(Path(path)):
        acquired.set()
        time.sleep(30)


def test_lock_timeout_and_process_crash_release(tmp_path):
    lock_path = tmp_path / ".campaign.yaml.lock"
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    worker = context.Process(
        target=_hold_lock_process, args=(str(lock_path), acquired),
    )
    worker.start()
    try:
        assert acquired.wait(20)
        with pytest.raises(TimeoutError, match="Timed out waiting for file lock"):
            with exclusive_file_lock(lock_path, timeout_seconds=0.1):
                pytest.fail("A second process entered the locked transaction")
        worker.terminate()
        worker.join(5)
        assert not worker.is_alive()
        with exclusive_file_lock(lock_path, timeout_seconds=1):
            assert lock_path.exists()
        # Keeping this stable file is intentional, including after a crash.
        assert lock_path.exists()
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(5)


@pytest.mark.parametrize("kind", ["state", "aux"])
def test_repeated_registration_is_idempotent(tmp_path, kind):
    path = tmp_path / "campaign.yaml"
    path.write_text("states: {}\naux: {}\n", encoding="utf-8")
    match = _match(path, kind=kind)
    _persist_automatic_measurement(match, "dataset")
    first = path.read_bytes()
    _persist_automatic_measurement(match, "dataset")
    assert path.read_bytes() == first


@pytest.mark.parametrize("kind,content", [
    ("state", "states: {s001: {data: other}}\n"),
    ("aux", "aux: {dataset: {state: s002}}\n"),
    ("state", "states: []\n"),
    ("aux", "aux: []\n"),
])
def test_invalid_or_conflicting_registration_preserves_yaml(tmp_path, kind, content):
    path = tmp_path / "campaign.yaml"
    path.write_text(content, encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError):
        _persist_automatic_measurement(_match(path, kind=kind), "dataset")
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_replacement_preserves_yaml_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "campaign.yaml"
    path.write_text("states: {}\naux: {}\n", encoding="utf-8")
    original = path.read_bytes()

    def fail_replace(self, target):
        raise OSError("replacement failed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="replacement failed"):
            _persist_automatic_measurement(_match(path), "dataset")
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    _persist_automatic_measurement(_match(path), "dataset")
    assert "dataset" in yaml.safe_load(path.read_text(encoding="utf-8"))["aux"]
