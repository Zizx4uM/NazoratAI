"""
FastAPI REST API application for NazoratAI.
Provides endpoints for portfolio querying, real-time ML risk scoring,
EWS alert evaluation, document auditing, and report generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from nazorat_ai.config import RiskTier
from nazorat_ai.data.generator import generate_synthetic_portfolio
from nazorat_ai.ews.rules import EWSEngine, evaluate_borrower_ews
from nazorat_ai.ml.features import FeaturePipeline
from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.ml.explainability import ExplainabilityEngine
from nazorat_ai.document_review.analyzer import HybridDocumentReviewer
from nazorat_ai.reporting.generator import ReportGenerator


class BorrowerInput(BaseModel):
    """Input payload for scoring an enterprise borrower."""
    inn: str = Field(..., example="305123456")
    company_name: str = Field("Yangi Sanoat MCHJ", example="Yangi Sanoat MCHJ")
    sector: str = Field("Tekstil va tikuvchilik", example="Tekstil va tikuvchilik")
    region: str = Field("Toshkent viloyati", example="Toshkent viloyati")
    branch_bxm: str = Field("Chirchiq BXM", example="Chirchiq BXM")
    officer_in_charge: str = Field("Akmal Rahimov", example="Akmal Rahimov")
    is_top20: bool = Field(False, example=False)
    loan_amount_uzs: float = Field(15000000000.0, example=15000000000.0)
    loan_balance_uzs: float = Field(12000000000.0, example=12000000000.0)
    interest_rate_pct: float = Field(21.0, example=21.0)
    collateral_value_uzs: float = Field(15000000000.0, example=15000000000.0)
    collateral_coverage_ratio: float = Field(1.25, example=1.25)
    revenue_current_uzs: float = Field(4000000000.0, example=4000000000.0)
    revenue_prev_month_uzs: float = Field(4200000000.0, example=4200000000.0)
    revenue_change_pct: float = Field(-0.047, example=-0.047)
    production_units_current: float = Field(45000.0, example=45000.0)
    production_units_prev: float = Field(46000.0, example=46000.0)
    production_change_pct: float = Field(-0.021, example=-0.021)
    electricity_kwh_current: float = Field(120000.0, example=120000.0)
    electricity_kwh_prev: float = Field(122000.0, example=122000.0)
    electricity_change_pct: float = Field(-0.016, example=-0.016)
    employee_count_current: int = Field(85, example=85)
    employee_count_prev: int = Field(85, example=85)
    ebitda_margin_pct: float = Field(16.5, example=16.5)
    net_profit_margin_pct: float = Field(9.0, example=9.0)
    current_ratio: float = Field(1.65, example=1.65)
    debt_to_equity: float = Field(1.40, example=1.40)
    dscr: float = Field(1.55, example=1.55)
    tax_arrears_uzs: float = Field(0.0, example=0.0)
    k2_card_index_amount_uzs: float = Field(0.0, example=0.0)
    overdue_days: int = Field(0, example=0)
    katm_credit_score: int = Field(420, example=420)


def create_app() -> FastAPI:
    """Application factory for NazoratAI REST API."""
    app = FastAPI(
        title="NazoratAI API",
        version="1.0.0",
        description="Banking Corporate Credit Monitoring and Early Warning System (EWS) REST API",
    )

    # In-memory cached portfolio and model instances
    portfolio_df: Optional[pd.DataFrame] = None
    credit_model: Optional[CreditRiskModel] = None
    ews_engine = EWSEngine()
    doc_reviewer = HybridDocumentReviewer()
    report_gen = ReportGenerator()

    def get_or_init_data_and_model():
        nonlocal portfolio_df, credit_model
        if portfolio_df is None:
            portfolio_df = generate_synthetic_portfolio(n_samples=603, seed=42)
        if credit_model is None:
            credit_model = CreditRiskModel()
            X = credit_model.feature_pipeline.extract_features(portfolio_df)
            y = portfolio_df["ground_truth_risk_tier"].tolist()
            credit_model.fit(X, y)
        return portfolio_df, credit_model

    @app.get("/health", tags=["System"])
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "NazoratAI",
            "version": "1.0.0",
            "banking_scope": "Uzbekistan Corporate Credit Monitoring",
        }

    @app.get("/api/v1/portfolio/summary", tags=["Portfolio"])
    def portfolio_summary():
        """Return aggregate statistics and risk tier breakdown for the 600+ corporate portfolio."""
        df, model = get_or_init_data_and_model()
        total_count = len(df)
        total_loan_amount = float(df["loan_amount_uzs"].sum())
        total_balance = float(df["loan_balance_uzs"].sum())

        tier_counts = df["ground_truth_risk_tier"].value_counts().to_dict()
        top20_df = df[df["is_top20"]]
        top20_balance = float(top20_df["loan_balance_uzs"].sum())

        overdue_45_count = int((df["overdue_days"] > 45).sum())
        overdue_90_count = int((df["overdue_days"] > 90).sum())
        k2_count = int((df["k2_card_index_amount_uzs"] > 0).sum())
        revenue_drop_count = int((df["revenue_change_pct"] <= -0.15).sum())

        return {
            "total_borrowers": total_count,
            "total_loan_amount_uzs": total_loan_amount,
            "total_outstanding_balance_uzs": total_balance,
            "top20_exposure_uzs": top20_balance,
            "top20_share_pct": round(top20_balance / max(total_balance, 1.0) * 100, 2),
            "risk_tier_breakdown": {
                "GREEN": int(tier_counts.get(RiskTier.GREEN.value, 0)),
                "YELLOW": int(tier_counts.get(RiskTier.YELLOW.value, 0)),
                "RED": int(tier_counts.get(RiskTier.RED.value, 0)),
            },
            "active_early_warning_signals": {
                "overdue_over_45_days": overdue_45_count,
                "overdue_over_90_days_npl": overdue_90_count,
                "k2_card_index_arrears": k2_count,
                "revenue_slump_over_15pct": revenue_drop_count,
            },
        }

    @app.get("/api/v1/portfolio/clients", tags=["Portfolio"])
    def list_clients(
        sector: Optional[str] = Query(None, description="Filter by industry sector"),
        region: Optional[str] = Query(None, description="Filter by administrative region"),
        risk_tier: Optional[str] = Query(None, description="Filter by GREEN, YELLOW, or RED"),
        is_top20: Optional[bool] = Query(None, description="Filter TOP-20 corporate borrowers"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        """List and filter corporate borrowers from portfolio."""
        df, _ = get_or_init_data_and_model()
        filtered = df.copy()

        if sector:
            filtered = filtered[filtered["sector"].str.contains(sector, case=False, na=False)]
        if region:
            filtered = filtered[filtered["region"].str.contains(region, case=False, na=False)]
        if risk_tier:
            filtered = filtered[filtered["ground_truth_risk_tier"] == risk_tier.upper()]
        if is_top20 is not None:
            filtered = filtered[filtered["is_top20"] == is_top20]

        total_matched = len(filtered)
        records = filtered.iloc[offset : offset + limit].to_dict(orient="records")

        return {
            "total_matched": total_matched,
            "limit": limit,
            "offset": offset,
            "clients": records,
        }

    @app.get("/api/v1/borrower/{inn}", tags=["Borrower 360"])
    def get_borrower_dossier(inn: str):
        """Retrieve complete dossier: profile, EWS alerts, ML score, and SHAP drivers for a single INN."""
        df, model = get_or_init_data_and_model()
        matched = df[df["inn"] == inn]
        if matched.empty:
            raise HTTPException(status_code=404, detail=f"Borrower with INN {inn} not found.")

        record = matched.iloc[0].to_dict()
        rule_result = ews_engine.evaluate_borrower(record)
        ml_scoring = model.evaluate_borrower_record(record)

        explainer = ExplainabilityEngine(model)
        explain_dict = explainer.explain_borrower_record(record)

        return {
            "profile": record,
            "ews_evaluation": {
                "risk_tier": rule_result.risk_tier,
                "total_alerts": rule_result.total_alerts,
                "critical_alerts_count": rule_result.critical_alerts_count,
                "high_alerts_count": rule_result.high_alerts_count,
                "action_summary": rule_result.recommended_action_summary,
                "alerts": [alert.__dict__ for alert in rule_result.alerts],
            },
            "ml_scoring": ml_scoring,
            "explainability": explain_dict,
        }

    @app.post("/api/v1/predict", tags=["Scoring & AI"])
    def predict_risk(borrower: BorrowerInput):
        """Real-time credit risk scoring and SHAP feature attribution for input parameters."""
        _, model = get_or_init_data_and_model()
        record = borrower.model_dump()
        rule_result = ews_engine.evaluate_borrower(record)
        ml_scoring = model.evaluate_borrower_record(record)

        explainer = ExplainabilityEngine(model)
        explain_dict = explainer.explain_borrower_record(record)

        return {
            "inn": borrower.inn,
            "company_name": borrower.company_name,
            "ews_rule_result": {
                "risk_tier": rule_result.risk_tier,
                "total_alerts": rule_result.total_alerts,
                "alerts": [a.__dict__ for a in rule_result.alerts],
                "recommended_action": rule_result.recommended_action_summary,
            },
            "ml_assessment": ml_scoring,
            "explainability": explain_dict,
        }

    @app.post("/api/v1/review-document", tags=["Document AI"])
    async def review_document(file: UploadFile = File(...)):
        """Upload and audit monitoring act (.docx, .pdf, .xlsx, .txt) with hybrid intelligence."""
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".docx", ".pdf", ".xlsx", ".xls", ".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {suffix}. Supported: .docx, .pdf, .xlsx, .txt"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            audit_result = doc_reviewer.review_file(tmp_path)
            return {
                "document_name": file.filename,
                "file_type": audit_result.file_type,
                "audit_mode": audit_result.audit_mode,
                "overall_risk_level": audit_result.overall_risk_level,
                "extracted_inn": audit_result.extracted_inn,
                "extracted_company": audit_result.extracted_company,
                "flags_count": audit_result.flags_count,
                "critical_flags_count": audit_result.critical_flags_count,
                "executive_summary": audit_result.executive_summary,
                "operational_status": audit_result.operational_status,
                "collateral_status": audit_result.collateral_status,
                "financial_health_status": audit_result.financial_health_status,
                "action_plan": audit_result.action_plan,
                "flags": [f.__dict__ for f in audit_result.flags],
            }
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @app.post("/api/v1/generate-report", tags=["Reporting"])
    def generate_report(borrower: BorrowerInput, format: str = Query("pdf", pattern="^(pdf|md)$")):
        """Generate and download formal PDF or Markdown Credit Monitoring Act."""
        _, model = get_or_init_data_and_model()
        record = borrower.model_dump()
        rule_result = ews_engine.evaluate_borrower(record)
        ml_scoring = model.evaluate_borrower_record(record)
        explainer = ExplainabilityEngine(model)
        explain_dict = explainer.explain_borrower_record(record)

        if format == "pdf":
            out_file = report_gen.generate_pdf_report(record, rule_result, ml_scoring, explain_dict)
            return FileResponse(
                path=out_file,
                media_type="application/pdf",
                filename=out_file.name,
            )
        else:
            out_file = report_gen.generate_markdown_report(record, rule_result, ml_scoring, explain_dict)
            return FileResponse(
                path=out_file,
                media_type="text/markdown",
                filename=out_file.name,
            )

    return app
