import pandas as pd

from deflector_tuning.analysis.sparameter_selection import select_s11_rows


def test_select_s11_rows_filters_named_sparameters_case_insensitively() -> None:
    table = pd.DataFrame(
        [
            {"s_name": "S11", "value": 1},
            {"s_name": "s11", "value": 2},
            {"s_name": "S21", "value": 3},
        ]
    )

    result = select_s11_rows(table)

    assert result["value"].tolist() == [1, 2]


def test_select_s11_rows_preserves_tables_without_s_name() -> None:
    table = pd.DataFrame([{"value": 1}])

    result = select_s11_rows(table)

    assert result.equals(table)
    assert result is not table
