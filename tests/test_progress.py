import os

from deflector_tuning.progress import progress_iter


def test_progress_iter_can_be_disabled_with_environment(monkeypatch) -> None:
    monkeypatch.setitem(os.environ, "DEFLECTOR_TUNING_PROGRESS", "0")

    assert list(progress_iter([1, 2, 3], desc="disabled", total=3)) == [1, 2, 3]
