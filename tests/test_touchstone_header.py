from deflector_tuning.data_loading.readers.touchstone_reader import TouchstoneHeader, parse_touchstone_header


def test_parse_ri_header_with_zero_reference() -> None:
    header = parse_touchstone_header("# GHz S RI R 0")

    assert header == TouchstoneHeader(
        frequency_unit="GHz",
        parameter="S",
        data_format="RI",
        reference_ohm=0.0,
        is_normalized=False,
    )


def test_nonzero_reference_means_normalized_data() -> None:
    header = parse_touchstone_header("# GHz S RI R 50")

    assert header.reference_ohm == 50.0
    assert header.is_normalized is True


def test_non_s_parameter_reference_does_not_mean_normalized_data() -> None:
    header = parse_touchstone_header("# GHz Z RI R 1")

    assert header.reference_ohm == 1.0
    assert header.is_normalized is False


def test_missing_format_defaults_to_ri() -> None:
    header = parse_touchstone_header("# GHz S R 50")

    assert header.data_format == "RI"
    assert header.reference_ohm == 50.0


def test_missing_reference_defaults_to_50_ohm() -> None:
    header = parse_touchstone_header("# GHz S RI")

    assert header.reference_ohm == 50.0


def test_header_parser_accepts_lowerprepro_260415_sweep_case_and_extra_spaces() -> None:
    header = parse_touchstone_header("  #   mhz   s   ri   r   50  ")

    assert header.frequency_unit == "MHz"
    assert header.parameter == "S"
    assert header.data_format == "RI"
    assert header.reference_ohm == 50.0
