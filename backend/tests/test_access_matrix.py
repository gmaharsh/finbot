from finbot.access_matrix import COLLECTION_ACCESS_ROLES, ROLE_COLLECTIONS, collections_for_role


def test_c_level_sees_all_collections():
    assert set(collections_for_role("c_level")) == {
        "general",
        "finance",
        "engineering",
        "marketing",
    }


def test_engineering_not_finance():
    assert "finance" not in collections_for_role("engineering")
    assert "engineering" in collections_for_role("engineering")


def test_finance_collection_roles():
    assert set(COLLECTION_ACCESS_ROLES["finance"]) == {"finance", "c_level"}


def test_general_open_to_all_roles():
    assert len(COLLECTION_ACCESS_ROLES["general"]) == 5
