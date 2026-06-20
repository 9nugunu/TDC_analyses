from pathlib import Path

from deflector_tuning.data_loading.touchstone_ports import write_s2p_with_ports_swapped


def test_write_s2p_with_ports_swapped_exchanges_assignments_and_sparameter_pairs(tmp_path: Path) -> None:
    source = tmp_path / "case.s2p"
    output = tmp_path / "swapped.s2p"
    source.write_text(
        "\n".join(
            [
                "! Touchstone port assignment:",
                '! Touchstone port 1 = CST MWS port 1 ("left")',
                '! Touchstone port 2 = CST MWS port 2 ("right")',
                "# GHz S RI R 0",
                "2.6 11 12 21 22 12 13 22 23",
            ]
        ),
        encoding="utf-8",
    )

    write_s2p_with_ports_swapped(source, output)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "! Touchstone port assignment:",
        '! Touchstone port 1 = CST MWS port 2 ("right")',
        '! Touchstone port 2 = CST MWS port 1 ("left")',
        "# GHz S RI R 0",
        "2.6 22 23 12 13 21 22 11 12",
    ]
