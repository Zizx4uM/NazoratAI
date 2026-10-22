"""
Tests for Early Warning System (EWS) trigger rules and alert engine.
"""

import pytest
from nazorat_ai.config import RiskTier, EWSSeverity
from nazorat_ai.ews.rules import EWSEngine, scan_portfolio_ews
from nazorat_ai.data.generator import generate_synthetic_portfolio


@pytest.fixture
def ews():
    return EWSEngine()


def test_healthy_borrower_no_alerts(ews):
    """Healthy borrower profile should generate zero alerts and GREEN tier."""
    record = {
        "inn": "300000001",
        "company_name": "Barkamol Agro MCHJ",
        "is_top20": False,
        "overdue_days": 0,
        "revenue_change_pct": 0.05,
        "production_change_pct": 0.03,
        "electricity_change_pct": 0.02,
        "collateral_coverage_ratio": 1.45,
        "dscr": 1.60,
        "k2_card_index_amount_uzs": 0.0,
    }
    result = ews.evaluate_borrower(record)
    assert result.total_alerts == 0
    assert result.risk_tier == RiskTier.GREEN.value
    assert "Standart monitoring" in result.recommended_action_summary


def test_overdue_boundary_conditions(ews):
    """Verify boundaries: 45 days (not yellow) vs 46 days (yellow), 90 days vs 91 days (red)."""
    # Exactly 45 days -> not yet yellow overdue trigger
    rec_45 = {"inn": "1", "overdue_days": 45, "collateral_coverage_ratio": 1.2, "dscr": 1.3}
    res_45 = ews.evaluate_borrower(rec_45)
    assert not any(a.rule_id == "EWS-02" for a in res_45.alerts)

    # 46 days -> Yellow emergency trigger (EWS-02)
    rec_46 = {"inn": "2", "overdue_days": 46, "collateral_coverage_ratio": 1.2, "dscr": 1.3}
    res_46 = ews.evaluate_borrower(rec_46)
    assert any(a.rule_id == "EWS-02" for a in res_46.alerts)
    assert res_46.risk_tier == RiskTier.YELLOW.value

    # Exactly 90 days -> Yellow trigger EWS-02 (not red EWS-01)
    rec_90 = {"inn": "3", "overdue_days": 90, "collateral_coverage_ratio": 1.2, "dscr": 1.3}
    res_90 = ews.evaluate_borrower(rec_90)
    assert any(a.rule_id == "EWS-02" for a in res_90.alerts)
    assert not any(a.rule_id == "EWS-01" for a in res_90.alerts)

    # 91 days -> Red trigger EWS-01
    rec_91 = {"inn": "4", "overdue_days": 91, "collateral_coverage_ratio": 1.2, "dscr": 1.3}
    res_91 = ews.evaluate_borrower(rec_91)
    assert any(a.rule_id == "EWS-01" for a in res_91.alerts)
    assert res_91.risk_tier == RiskTier.RED.value


def test_overdue_45_days_yellow_trigger(ews):
    """Overdue > 45 days triggers EWS-02 and emergency monitoring act requirement."""
    record = {
        "inn": "300000002",
        "company_name": "Sariq Toifa Klaster MCHJ",
        "is_top20": False,
        "overdue_days": 52,
        "revenue_change_pct": -0.05,
        "production_change_pct": -0.05,
        "collateral_coverage_ratio": 1.10,
        "dscr": 1.25,
        "k2_card_index_amount_uzs": 0.0,
    }
    result = ews.evaluate_borrower(record)
    assert result.risk_tier == RiskTier.YELLOW.value
    rule_ids = [a.rule_id for a in result.alerts]
    assert "EWS-02" in rule_ids
    assert any("Favqulodda monitoring" in a.recommended_action for a in result.alerts)


def test_overdue_90_days_red_npl_trigger(ews):
    """Overdue > 90 days triggers EWS-01 and RED / NPL status."""
    record = {
        "inn": "300000003",
        "company_name": "Qizil Toifa Zavod AJ",
        "is_top20": False,
        "overdue_days": 110,
        "revenue_change_pct": -0.20,
        "production_change_pct": -0.25,
        "collateral_coverage_ratio": 0.85,
        "dscr": 0.65,
        "k2_card_index_amount_uzs": 500e6,
    }
    result = ews.evaluate_borrower(record)
    assert result.risk_tier == RiskTier.RED.value
    rule_ids = [a.rule_id for a in result.alerts]
    assert "EWS-01" in rule_ids


def test_revenue_and_production_drop_triggers(ews):
    """Revenue drop >= 15% (EWS-03) and production drop >= 20% (EWS-04)."""
    record = {
        "inn": "300000004",
        "company_name": "Pasayuvchi Korxona XK",
        "is_top20": False,
        "overdue_days": 10,
        "revenue_change_pct": -0.18,      # Trigger EWS-03
        "production_change_pct": -0.25,   # Trigger EWS-04
        "collateral_coverage_ratio": 1.30,
        "dscr": 1.35,
        "k2_card_index_amount_uzs": 0.0,
    }
    result = ews.evaluate_borrower(record)
    rule_ids = [a.rule_id for a in result.alerts]
    assert "EWS-03" in rule_ids
    assert "EWS-04" in rule_ids


def test_k2_and_collateral_triggers(ews):
    """Card Index 2 arrears (EWS-05) and under-collateralized (EWS-07)."""
    record = {
        "inn": "300000005",
        "company_name": "Garov Yetishmovchi MCHJ",
        "is_top20": False,
        "overdue_days": 0,
        "revenue_change_pct": 0.02,
        "production_change_pct": 0.01,
        "collateral_coverage_ratio": 0.88,  # Under 1.0 -> EWS-07
        "dscr": 1.40,
        "k2_card_index_amount_uzs": 150e6, # K-2 arrears -> EWS-05
    }
    result = ews.evaluate_borrower(record)
    rule_ids = [a.rule_id for a in result.alerts]
    assert "EWS-05" in rule_ids
    assert "EWS-07" in rule_ids


def test_top20_escalation_alert(ews):
    """TOP-20 borrower with any alert must trigger EWS-09 board escalation."""
    record = {
        "inn": "300000006",
        "company_name": "Yirik Klaster TOP-20 AJ",
        "is_top20": True,
        "overdue_days": 46,
        "revenue_change_pct": -0.16,
        "production_change_pct": -0.10,
        "collateral_coverage_ratio": 1.20,
        "dscr": 1.15,
        "k2_card_index_amount_uzs": 0.0,
    }
    result = ews.evaluate_borrower(record)
    assert result.is_top20 is True
    rule_ids = [a.rule_id for a in result.alerts]
    assert "EWS-09" in rule_ids


def test_portfolio_scan():
    """Verify batch scan of entire synthetic portfolio."""
    df = generate_synthetic_portfolio(n_samples=50, seed=42)
    results = scan_portfolio_ews(df)
    assert len(results) == 50
    assert all(r.risk_tier in (RiskTier.GREEN.value, RiskTier.YELLOW.value, RiskTier.RED.value) for r in results)
