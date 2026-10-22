"""
NazoratAI CLI tool: provides terminal commands for data generation,
model training, portfolio scanning, single borrower audit, and server execution.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from nazorat_ai.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_REPORT_DIR,
    PORTFOLIO_TOTAL_BORROWERS,
)
from nazorat_ai.data.generator import generate_synthetic_portfolio
from nazorat_ai.data.loader import load_portfolio_data
from nazorat_ai.ews.rules import EWSEngine, scan_portfolio_ews
from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.ml.explainability import ExplainabilityEngine
from nazorat_ai.ml.evaluator import evaluate_model_performance
from nazorat_ai.document_review.analyzer import HybridDocumentReviewer
from nazorat_ai.reporting.generator import ReportGenerator, generate_portfolio_summary_report


def cmd_generate_data(args):
    """Generate synthetic portfolio dataset."""
    output_path = Path(args.output) if args.output else DEFAULT_DATA_DIR / "corporate_portfolio.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating realistic enterprise portfolio dataset with {args.count} borrowers...")
    df = generate_synthetic_portfolio(n_samples=args.count, seed=args.seed)

    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"✅ Successfully generated {len(df)} records saved to: {output_path}")
    print(f"   - Green tier: {(df['ground_truth_risk_tier'] == 'GREEN').sum()}")
    print(f"   - Yellow tier: {(df['ground_truth_risk_tier'] == 'YELLOW').sum()}")
    print(f"   - Red tier: {(df['ground_truth_risk_tier'] == 'RED').sum()}")
    print(f"   - TOP-20 borrowers: {df['is_top20'].sum()}")


def cmd_train_model(args):
    """Train ML Credit Risk Model with 80/20 stratified split and 5-fold cross validation."""
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from nazorat_ai.ml.model import LABEL_TO_INT

    data_path = Path(args.data) if args.data else DEFAULT_DATA_DIR / "corporate_portfolio.csv"
    if not data_path.exists():
        print(f"Data file not found at {data_path}. Generating default dataset first...")
        df = generate_synthetic_portfolio(n_samples=PORTFOLIO_TOTAL_BORROWERS)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = load_portfolio_data(data_path)

    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "credit_risk_model.joblib"

    print(f"Training CreditRiskModel on {len(df)} corporate borrower records...")
    model = CreditRiskModel()
    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()

    # 1. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Fit on training split & evaluate on held-out test set
    model.fit(X_train, y_train)
    metrics = evaluate_model_performance(model, X_test, y_test)

    # 3. 5-Fold Stratified Cross-Validation on full feature space
    y_encoded = [LABEL_TO_INT[val] for val in y]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model.classifier, X, y_encoded, cv=cv, scoring="f1_macro")
    cv_roc = cross_val_score(model.classifier, X, y_encoded, cv=cv, scoring="roc_auc_ovr")

    # 4. Fit on full data for production deployment artifact
    model.fit(X, y)
    model.save(model_path)
    print(f"✅ Model trained and saved to: {model_path}")
    print("\n📈 Held-Out Test Evaluation (20% sample, N={}):".format(len(y_test)))
    print(f"   - Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"   - Test Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   - Test ROC-AUC (OVR): {metrics['roc_auc_ovr']:.4f}")
    print(f"   - EWS Watchlist Precision: {metrics['ews_watch_precision'] * 100:.2f}% (Target: >=70%)")
    print(f"   - EWS Watchlist Recall: {metrics['ews_watch_recall'] * 100:.2f}% (Target: >=65%)")
    print("\n🔬 5-Fold Stratified Cross-Validation:")
    print(f"   - Macro F1: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
    print(f"   - ROC-AUC:  {cv_roc.mean():.4f} (±{cv_roc.std():.4f})")


def cmd_scan_portfolio(args):
    """Scan portfolio with EWS rules engine."""
    data_path = Path(args.data) if args.data else DEFAULT_DATA_DIR / "corporate_portfolio.csv"
    if not data_path.exists():
        df = generate_synthetic_portfolio(n_samples=PORTFOLIO_TOTAL_BORROWERS)
    else:
        df = load_portfolio_data(data_path)

    print(f"Scanning {len(df)} corporate borrowers for Early Warning System (EWS) triggers...")
    scan_results = scan_portfolio_ews(df)

    green_count = sum(1 for r in scan_results if r.risk_tier == "GREEN")
    yellow_count = sum(1 for r in scan_results if r.risk_tier == "YELLOW")
    red_count = sum(1 for r in scan_results if r.risk_tier == "RED")
    total_alerts = sum(r.total_alerts for r in scan_results)

    print("📊 Portfolio Scan Results:")
    print(f"   - 🟢 Green (Healthy / Standard): {green_count} ({green_count/max(len(df), 1)*100:.1f}%)")
    print(f"   - 🟡 Yellow (Watchlist / Overdue >45d): {yellow_count} ({yellow_count/max(len(df), 1)*100:.1f}%)")
    print(f"   - 🔴 Red (Distressed / NPL >90d): {red_count} ({red_count/max(len(df), 1)*100:.1f}%)")
    print(f"   - Total Active EWS Alerts: {total_alerts}")

    summary_file = generate_portfolio_summary_report(df, scan_results, args.report_dir)
    print(f"✅ Executive summary saved to: {summary_file}")


def cmd_audit_borrower(args):
    """Audit single borrower by INN."""
    data_path = Path(args.data) if args.data else DEFAULT_DATA_DIR / "corporate_portfolio.csv"
    if not data_path.exists():
        df = generate_synthetic_portfolio(n_samples=PORTFOLIO_TOTAL_BORROWERS)
    else:
        df = load_portfolio_data(data_path)

    matched = df[df["inn"] == str(args.inn)]
    if matched.empty:
        print(f"❌ Error: Borrower with INN {args.inn} not found in portfolio.")
        sys.exit(1)

    record = matched.iloc[0].to_dict()
    ews_engine = EWSEngine()
    rule_res = ews_engine.evaluate_borrower(record)

    model = CreditRiskModel()
    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()
    model.fit(X, y)

    ml_eval = model.evaluate_borrower_record(record)
    explainer = ExplainabilityEngine(model)
    explain_res = explainer.explain_borrower_record(record)

    print("=" * 60)
    print(f"AUDIT DOSSIER: {record['company_name']} (INN: {record['inn']})")
    print("=" * 60)
    print(f"Soha: {record['sector']} | Hudud: {record['region']} | BXM: {record['branch_bxm']}")
    print(f"Kredit qoldig'i: {record['loan_balance_uzs']:,.0f} UZS | Garov qoplami: {record['collateral_coverage_ratio']:.2f}x")
    print(f"Muddati o'tgan: {record['overdue_days']} kun | 2-Kartoteka: {record['k2_card_index_amount_uzs']:,.0f} UZS")
    print(f"Tushum MoM: {record['revenue_change_pct']*100:.1f}% | Ishlab chiqarish MoM: {record['production_change_pct']*100:.1f}%")
    print("-" * 60)
    print(f"EWS Risk Tier: {rule_res.risk_tier} | Jami EWS signallari: {rule_res.total_alerts} ta")
    for a in rule_res.alerts:
        print(f"  * [{a.rule_id}] {a.rule_name} ({a.severity}): {a.observed_value}")
    print("-" * 60)
    print(f"ML Predicted Tier: {ml_eval['predicted_risk_tier']} | ML Score: {ml_eval['risk_score_0_100']}/100")
    print(f"SHAP Xulosa: {explain_res['executive_summary_uz']}")
    print("=" * 60)

    report_gen = ReportGenerator(args.report_dir)
    md_file = report_gen.generate_markdown_report(record, rule_res, ml_eval, explain_res)
    print(f"✅ Markdown Dalolatnoma saqlandi: {md_file}")

    if args.pdf:
        pdf_file = report_gen.generate_pdf_report(record, rule_res, ml_eval, explain_res)
        print(f"✅ PDF Dalolatnoma saqlandi: {pdf_file}")


def cmd_review_doc(args):
    """Audit monitoring document file (.docx, .pdf, .xlsx, .txt)."""
    path = Path(args.file)
    if not path.exists():
        print(f"❌ Error: File not found at {path}")
        sys.exit(1)

    print(f"Reviewing monitoring document: {path.name}...")
    reviewer = HybridDocumentReviewer()
    result = reviewer.review_file(path)

    print("=" * 60)
    print(f"DOCUMENT AUDIT RESULT: {result.document_name}")
    print("=" * 60)
    print(f"Audit Mode: {result.audit_mode}")
    print(f"Overall Risk Level: {result.overall_risk_level}")
    print(f"Extracted INN: {result.extracted_inn or 'N/A'}")
    print(f"Extracted Company: {result.extracted_company or 'N/A'}")
    print(f"Summary: {result.executive_summary}")
    print(f"Operational Status: {result.operational_status}")
    print(f"Collateral Status: {result.collateral_status}")
    print(f"Financial Health: {result.financial_health_status}")
    print("-" * 60)
    print(f"Flags Detected ({len(result.flags)}):")
    for f in result.flags:
        print(f"  * [{f.flag_id}] {f.category} ({f.severity})")
        print(f"    Snippet: {f.context_snippet}")
        print(f"    Recommendation: {f.recommendation}")
    print("=" * 60)


def cmd_serve_api(args):
    """Launch FastAPI server via uvicorn."""
    import uvicorn
    from nazorat_ai.api.app import create_app
    print(f"Starting NazoratAI REST API on http://{args.host}:{args.port}...")
    uvicorn.run("nazorat_ai.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


def cmd_run_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess
    dash_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    print(f"Starting NazoratAI Web Dashboard on http://localhost:{args.port}...")
    subprocess.run(["streamlit", "run", str(dash_path), "--server.port", str(args.port)])


def app():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nazorat-ai",
        description="NazoratAI: Banking Corporate Credit Monitoring and Early Warning System (EWS)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate synthetic corporate credit dataset")
    p_gen.add_argument("--count", type=int, default=PORTFOLIO_TOTAL_BORROWERS, help="Number of borrowers")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--output", type=str, default=None, help="Output file path (.csv or .xlsx)")
    p_gen.set_defaults(func=cmd_generate_data)

    # train-model
    p_train = subparsers.add_parser("train-model", help="Train ML credit risk classifier")
    p_train.add_argument("--data", type=str, default=None, help="Path to portfolio dataset")
    p_train.add_argument("--model-dir", type=str, default=None, help="Directory to save model artifact")
    p_train.set_defaults(func=cmd_train_model)

    # scan-portfolio
    p_scan = subparsers.add_parser("scan-portfolio", help="Run EWS scan across entire portfolio")
    p_scan.add_argument("--data", type=str, default=None, help="Path to portfolio dataset")
    p_scan.add_argument("--report-dir", type=str, default=None, help="Directory for summary reports")
    p_scan.set_defaults(func=cmd_scan_portfolio)

    # audit-borrower
    p_audit = subparsers.add_parser("audit-borrower", help="Audit single borrower by INN")
    p_audit.add_argument("--inn", type=str, required=True, help="Borrower INN (9 digits)")
    p_audit.add_argument("--data", type=str, default=None, help="Path to portfolio dataset")
    p_audit.add_argument("--report-dir", type=str, default=None, help="Directory for generated reports")
    p_audit.add_argument("--pdf", action="store_true", help="Generate PDF report in addition to Markdown")
    p_audit.set_defaults(func=cmd_audit_borrower)

    # review-doc
    p_doc = subparsers.add_parser("review-doc", help="Audit a monitoring act document (.docx, .pdf, .xlsx, .txt)")
    p_doc.add_argument("--file", type=str, required=True, help="Path to document file")
    p_doc.set_defaults(func=cmd_review_doc)

    # serve-api
    p_api = subparsers.add_parser("serve-api", help="Start FastAPI REST server")
    p_api.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    p_api.add_argument("--port", type=int, default=8000, help="Port")
    p_api.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_api.set_defaults(func=cmd_serve_api)

    # run-dashboard
    p_dash = subparsers.add_parser("run-dashboard", help="Start Streamlit web dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Dashboard port")
    p_dash.set_defaults(func=cmd_run_dashboard)

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    app()
