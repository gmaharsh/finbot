from finbot.chat_service import _intersect_collections
from finbot.router_semantic import ROUTE_ENGINEERING, ROUTE_FINANCE, ROUTE_HR_GENERAL


def test_finance_user_engineering_route_empty():
    targeted, user_cols = _intersect_collections(ROUTE_ENGINEERING, "finance")
    assert targeted == []
    assert "finance" in user_cols


def test_engineering_user_finance_route_empty():
    targeted, _ = _intersect_collections(ROUTE_FINANCE, "engineering")
    assert targeted == []


def test_engineering_user_hr_route_general_only():
    targeted, _ = _intersect_collections(ROUTE_HR_GENERAL, "engineering")
    assert targeted == ["general"]


def test_c_level_finance_route():
    targeted, _ = _intersect_collections(ROUTE_FINANCE, "c_level")
    assert targeted == ["finance"]
