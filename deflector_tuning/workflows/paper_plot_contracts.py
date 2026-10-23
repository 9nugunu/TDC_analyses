"""Validate the supported publication layout before creating output artifacts."""

MARKER_LABELS = {"f_2pi3": r"$f_{2\pi/3}$", "f_mean": r"$f_m$", "f_pi2": r"$f_{\pi/2}$"}


def marker_labels(config: dict) -> list[str]:
    markers = config["markers"]
    names = [item["name"] for item in markers]
    if len(names) != 3 or set(names) != set(MARKER_LABELS):
        raise ValueError("Publication layout requires exactly f_2pi3, f_mean and f_pi2")
    return [item.get("label", MARKER_LABELS[item["name"]]) for item in markers]


def validate_comparison_layout(config: dict) -> None:
    """Two cases and two pairs are the supported paper comparison layout."""
    marker_labels(config)
    for key, identity in (("cases", "id"), ("pairs", "name")):
        items = config[key]
        if len(items) != 2 or len({item[identity] for item in items}) != 2:
            raise ValueError(f"Publication layout requires exactly two distinct {key}")


def case_label(case: dict) -> str:
    return case.get("display_label", case["label"])
