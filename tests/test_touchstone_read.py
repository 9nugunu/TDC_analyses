from pathlib import Path

import pytest

from deflector_tuning.data_loading.readers.touchstone_reader import TouchstoneHeader, read_touchstone


def test_read_s1p_ri_file(tmp_path: Path) -> None:
    path = tmp_path / "case.s1p"
    path.write_text(
        "\n".join(
            [
                "! comment line",
                "# GHz S RI R 0",
                "2.6 0.8 -0.1",
                "2.7 0.7 -0.2",
            ]
        ),
        encoding="utf-8",
    )

    data = read_touchstone(path)

    assert data.header == TouchstoneHeader(
        frequency_unit="GHz",
        parameter="S",
        data_format="RI",
        reference_ohm=0.0,
        is_normalized=False,
    )
    assert data.frequency == [2.6, 2.7]
    assert data.s_values == [[0.8 - 0.1j], [0.7 - 0.2j]]


def test_read_s2p_ri_file_groups_four_complex_values_per_frequency(tmp_path: Path) -> None:
    path = tmp_path / "case.s2p"
    path.write_text(
        "\n".join(
            [
                "# GHz S RI R 50",
                "2.6 1 0 2 0 3 0 4 0",
            ]
        ),
        encoding="utf-8",
    )

    data = read_touchstone(path)

    assert data.header.is_normalized is True
    assert data.frequency == [2.6]
    assert data.s_values == [[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]]


def test_read_touchstone_rejects_non_ri_for_now(tmp_path: Path) -> None:
    path = tmp_path / "case.s1p"
    path.write_text("# GHz S MA R 50\n2.6 1 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Only RI Touchstone data is supported"):
        read_touchstone(path)
