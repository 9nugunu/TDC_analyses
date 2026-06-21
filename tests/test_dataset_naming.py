from deflector_tuning.data_loading.dataset_naming import parse_dataset_id


def test_parse_dataset_id_accepts_profile_category() -> None:
    identity = parse_dataset_id("sim_profile_260618_hem11_eh_zcut")

    assert identity.layer == "sim"
    assert identity.category == "profile"
    assert identity.date == "260618"
    assert identity.tail == "hem11_eh_zcut"
