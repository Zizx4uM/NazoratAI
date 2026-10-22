"""
Tests for automated Markdown and PDF report generation.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from nazorat_ai.data.generator import generate_synthetic_portfolio
from nazorat_ai.ews.rules import EWSEngine
from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.ml.explainability import ExplainabilityEngine
from nazorat_ai.reporting.generator import (
    ReportGenerator,
    generate_borrower_markdown_report,
    generate_borrower_pdf_report,
    generate_portfolio_summary_report,
)


@pytest.fixture(scope="module")
def report_data():
    df = generate_synthetic_portfolio(n_samples=30, seed=42)
    record = df.iloc[0].to_dict()

    ews_engine = EWSEngine()
    rule_res = ews_engine.evaluate_borrower(record)

    model = CreditRiskModel()
    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()
    model.fit(X, y)

    ml_eval = model.evaluate_borrower_record(record)
    explainer = ExplainabilityEngine(model)
    explain_res = explainer.explain_borrower_record(record)

    return record, rule_res, ml_eval, explain_res, df


def test_markdown_report_generation(report_data):
    """Verify Markdown audit act generation and contents."""
    record, rule_res, ml_eval, explain_res, _ = report_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = ReportGenerator(output_dir=tmp_dir)
        md_path = gen.generate_markdown_report(record, rule_res, ml_eval, explain_res)

        assert md_path.exists()
        assert md_path.stat().st_size > 500

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        assert record["inn"] in text
        assert "KREDIT MONITORINGI" in text
        assert "EWS" in text
        assert "SHAP" in text
        assert "Monitoring xodimi" in text


def test_markdown_report_without_explainability(report_data):
    """Verify Markdown report still compiles cleanly when explain_dict is None."""
    record, rule_res, ml_eval, _, _ = report_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = ReportGenerator(output_dir=tmp_dir)
        md_path = gen.generate_markdown_report(record, rule_res, ml_eval, explain_dict=None)
        assert md_path.exists()
        assert md_path.stat().st_size > 400


def test_pdf_report_generation_with_shap(report_data):
    """Verify PDF credit monitoring act generation via ReportLab with SHAP section."""
    record, rule_res, ml_eval, explain_res, _ = report_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = ReportGenerator(output_dir=tmp_dir)
        pdf_path = gen.generate_pdf_report(record, rule_res, ml_eval, explain_res)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000

        # Check PDF magic bytes (%PDF)
        with open(pdf_path, "rb") as f:
            header = f.read(4)
        assert header == b"%PDF"


def test_pdf_report_generation_without_shap(report_data):
    """Verify PDF report builds cleanly when explain_dict is None."""
    record, rule_res, ml_eval, _, _ = report_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = ReportGenerator(output_dir=tmp_dir)
        pdf_path = gen.generate_pdf_report(record, rule_res, ml_eval, explain_dict=None)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000


def test_portfolio_summary_report_generation(report_data):
    """Verify executive summary report for board."""
    _, _, _, _, df = report_data
    ews = EWSEngine()
    scan_results = [ews.evaluate_borrower(row.to_dict()) for _, row in df.iterrows()]

    with tempfile.TemporaryDirectory() as tmp_dir:
        summary_path = generate_portfolio_summary_report(df, scan_results, output_dir=tmp_dir)
        assert summary_path.exists()

        with open(summary_path, "r", encoding="utf-8") as f:
            text = f.read()

        assert "KREDIT PORTFELI VA EWS IJRO XULOSASI" in text
        assert str(len(df)) in text


def test_portfolio_summary_report_empty_df():
    """Verify executive summary report does not divide by zero on empty DataFrame."""
    empty_df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary_path = generate_portfolio_summary_report(empty_df, [], output_dir=tmp_dir)
        assert summary_path.exists()
        with open(summary_path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "0 ta korporativ" in text
