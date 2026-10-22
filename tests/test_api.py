"""
Integration tests for FastAPI REST API endpoints using TestClient.
"""

import io
import pytest
from fastapi.testclient import TestClient

from nazorat_ai.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "NazoratAI"


def test_portfolio_summary_endpoint(client):
    """Verify /api/v1/portfolio/summary returns portfolio breakdown."""
    response = client.get("/api/v1/portfolio/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_borrowers"] == 603
    assert "risk_tier_breakdown" in data
    assert "active_early_warning_signals" in data
    assert data["top20_exposure_uzs"] > 0


def test_portfolio_clients_filtering(client):
    """Verify /api/v1/portfolio/clients list and filtering."""
    response = client.get("/api/v1/portfolio/clients?limit=10&is_top20=true")
    assert response.status_code == 200
    data = response.json()
    assert len(data["clients"]) <= 10
    assert all(c["is_top20"] is True for c in data["clients"])


def test_borrower_dossier_endpoint(client):
    """Verify /api/v1/borrower/{inn} returns complete profile and scoring."""
    # First get a valid INN from the clients endpoint
    res_list = client.get("/api/v1/portfolio/clients?limit=1")
    valid_inn = res_list.json()["clients"][0]["inn"]

    response = client.get(f"/api/v1/borrower/{valid_inn}")
    assert response.status_code == 200
    data = response.json()
    assert data["profile"]["inn"] == valid_inn
    assert "ews_evaluation" in data
    assert "ml_scoring" in data
    assert "explainability" in data


def test_borrower_dossier_not_found(client):
    """Verify 404 for unknown INN."""
    response = client.get("/api/v1/borrower/999999999")
    assert response.status_code == 404


def test_predict_endpoint(client):
    """Verify POST /api/v1/predict risk scoring."""
    payload = {
        "inn": "309999888",
        "company_name": "Test Klaster MCHJ",
        "sector": "Tekstil va tikuvchilik",
        "region": "Toshkent shahri",
        "branch_bxm": "Yunusobod BXM",
        "officer_in_charge": "Bobur Alimov",
        "is_top20": False,
        "loan_amount_uzs": 10000000000.0,
        "loan_balance_uzs": 8000000000.0,
        "interest_rate_pct": 21.0,
        "collateral_value_uzs": 12000000000.0,
        "collateral_coverage_ratio": 1.5,
        "revenue_current_uzs": 3000000000.0,
        "revenue_prev_month_uzs": 3200000000.0,
        "revenue_change_pct": -0.062,
        "production_units_current": 10000.0,
        "production_units_prev": 10500.0,
        "production_change_pct": -0.047,
        "electricity_kwh_current": 50000.0,
        "electricity_kwh_prev": 51000.0,
        "electricity_change_pct": -0.02,
        "employee_count_current": 40,
        "employee_count_prev": 40,
        "ebitda_margin_pct": 18.0,
        "net_profit_margin_pct": 10.0,
        "current_ratio": 1.7,
        "debt_to_equity": 1.2,
        "dscr": 1.6,
        "tax_arrears_uzs": 0.0,
        "k2_card_index_amount_uzs": 0.0,
        "overdue_days": 0,
        "katm_credit_score": 440,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "ews_rule_result" in data
    assert "ml_assessment" in data
    assert "explainability" in data
    assert data["ml_assessment"]["predicted_risk_tier"] in ("GREEN", "YELLOW", "RED")


def test_review_document_endpoint(client):
    """Verify POST /api/v1/review-document multipart upload audit."""
    doc_text = "MONITORING DALOLATNOMASI\nKorxona: 'Sinov' MCHJ\nINN: 301234567\nFaoliyat to'liq to'xtatilgan."
    file_bytes = io.BytesIO(doc_text.encode("utf-8"))

    response = client.post(
        "/api/v1/review-document",
        files={"file": ("monitoring_dalolatnoma.txt", file_bytes, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["overall_risk_level"] == "RED"
    assert data["flags_count"] >= 1


def test_generate_report_endpoint_pdf(client):
    """Verify POST /api/v1/generate-report?format=pdf returns PDF stream."""
    payload = {
        "inn": "301112223",
        "company_name": "Hisobot MCHJ",
        "sector": "Oziq-ovqat sanoati",
        "region": "Samarqand viloyati",
        "branch_bxm": "Samarqand shahar BXM",
        "officer_in_charge": "Sanjar Yusupov",
        "is_top20": False,
        "loan_amount_uzs": 8000000000.0,
        "loan_balance_uzs": 6500000000.0,
        "interest_rate_pct": 20.0,
        "collateral_value_uzs": 9000000000.0,
        "collateral_coverage_ratio": 1.38,
        "revenue_current_uzs": 2500000000.0,
        "revenue_prev_month_uzs": 2600000000.0,
        "revenue_change_pct": -0.038,
        "production_units_current": 8000.0,
        "production_units_prev": 8200.0,
        "production_change_pct": -0.024,
        "electricity_kwh_current": 35000.0,
        "electricity_kwh_prev": 36000.0,
        "electricity_change_pct": -0.028,
        "employee_count_current": 35,
        "employee_count_prev": 35,
        "ebitda_margin_pct": 16.0,
        "net_profit_margin_pct": 9.5,
        "current_ratio": 1.5,
        "debt_to_equity": 1.3,
        "dscr": 1.5,
        "tax_arrears_uzs": 0.0,
        "k2_card_index_amount_uzs": 0.0,
        "overdue_days": 0,
        "katm_credit_score": 430,
    }
    response = client.post("/api/v1/generate-report?format=pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_generate_report_endpoint_markdown(client):
    """Verify POST /api/v1/generate-report?format=md returns Markdown stream."""
    payload = {
        "inn": "301112224",
        "company_name": "Hisobot Markdown MCHJ",
        "sector": "Tekstil va tikuvchilik",
        "region": "Andijon viloyati",
        "branch_bxm": "Andijon BXM",
        "officer_in_charge": "Bobur Alimov",
        "is_top20": False,
        "loan_amount_uzs": 5000000000.0,
        "loan_balance_uzs": 4000000000.0,
        "interest_rate_pct": 21.0,
        "collateral_value_uzs": 6000000000.0,
        "collateral_coverage_ratio": 1.5,
        "revenue_current_uzs": 1500000000.0,
        "revenue_prev_month_uzs": 1600000000.0,
        "revenue_change_pct": -0.05,
        "production_units_current": 5000.0,
        "production_units_prev": 5100.0,
        "production_change_pct": -0.02,
        "electricity_kwh_current": 20000.0,
        "electricity_kwh_prev": 20500.0,
        "electricity_change_pct": -0.02,
        "employee_count_current": 25,
        "employee_count_prev": 25,
        "ebitda_margin_pct": 15.0,
        "net_profit_margin_pct": 8.0,
        "current_ratio": 1.6,
        "debt_to_equity": 1.1,
        "dscr": 1.6,
        "tax_arrears_uzs": 0.0,
        "k2_card_index_amount_uzs": 0.0,
        "overdue_days": 0,
        "katm_credit_score": 420,
    }
    response = client.post("/api/v1/generate-report?format=md", json=payload)
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert b"KREDIT MONITORINGI" in response.content
