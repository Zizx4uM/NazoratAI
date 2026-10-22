"""
NazoratAI: Interactive Streamlit Web Dashboard for Banking Corporate Credit Monitoring.
Visualizes portfolio risk, TOP-20 watch-list, EWS early warning triggers,
client 360 dossier with ML SHAP explainability, and Document AI audit.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

# Ensure package is importable
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nazorat_ai.config import RiskTier, SECTORS, REGIONS
from nazorat_ai.data.generator import generate_synthetic_portfolio
from nazorat_ai.ews.rules import EWSEngine, scan_portfolio_ews
from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.ml.explainability import ExplainabilityEngine
from nazorat_ai.ml.evaluator import evaluate_model_performance
from nazorat_ai.document_review.analyzer import HybridDocumentReviewer
from nazorat_ai.reporting.generator import ReportGenerator


st.set_page_config(
    page_title="NazoratAI — Korporativ Kredit Monitoringi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_portfolio_data():
    """Generate or load cached 603-enterprise portfolio."""
    return generate_synthetic_portfolio(n_samples=603, seed=42)


@st.cache_resource
def get_trained_model(df: pd.DataFrame):
    """Train and cache the ML Credit Risk Model."""
    model = CreditRiskModel()
    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()
    model.fit(X, y)
    return model


df = get_portfolio_data()
model = get_trained_model(df)
ews_engine = EWSEngine()
doc_reviewer = HybridDocumentReviewer()
report_gen = ReportGenerator()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("NazoratAI")
st.sidebar.caption("Bank Korporativ Kredit Monitoringi & EWS")

menu = st.sidebar.radio(
    "Bo'limni tanlang:",
    [
        "📊 Portfel Umumiy Tahlili",
        "🏆 TOP-20 Yirik Qarz Oluvchilar",
        "⚠️ Erta Ogohlantirish (EWS)",
        "🔍 Qarz Oluvchi 360° & SHAP",
        "📑 Hujjatlar Auditi (Document AI)",
        "⚙️ ML Boshqaruvi & Metrikalar",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Qamrov:** 603 korporativ mijoz\n\n"
    "**Sohalar:** 10 ta asosiy sanoat tarmog'i\n\n"
    "**Tarkib:** 18 bosh ofis + 72 BXM xodimlari"
)

# -------------------------------------------------------------
# PAGE 1: PORTFOLIO EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if menu == "📊 Portfel Umumiy Tahlili":
    st.title("📊 Bank Kredit Portfeli — Ijro Tahlili")
    st.markdown("Kredit monitoringi departamenti va Bank Boshqaruvi uchun umumiy risk holati")

    total_borrowers = len(df)
    total_loans = df["loan_amount_uzs"].sum()
    total_balance = df["loan_balance_uzs"].sum()

    green_count = (df["ground_truth_risk_tier"] == RiskTier.GREEN.value).sum()
    yellow_count = (df["ground_truth_risk_tier"] == RiskTier.YELLOW.value).sum()
    red_count = (df["ground_truth_risk_tier"] == RiskTier.RED.value).sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Jami Qarz Oluvchilar", f"{total_borrowers} ta")
    col2.metric("Jami Portfel Qoldig'i", f"{total_balance / 1e12:.2f} trln UZS")
    col3.metric("🟢 Yashil (Standart)", f"{green_count} ta", f"{green_count / total_borrowers * 100:.1f}%")
    col4.metric("🟡 Sariq (Kuzatuv / EWS)", f"{yellow_count} ta", f"{yellow_count / total_borrowers * 100:.1f}%", delta_color="inverse")
    col5.metric("🔴 Qizil (Substandart)", f"{red_count} ta", f"{red_count / total_borrowers * 100:.1f}%", delta_color="inverse")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sohalar bo'yicha kredit qoldig'i (mlrd UZS)")
        sector_balance = df.groupby("sector")["loan_balance_uzs"].sum() / 1e9
        st.bar_chart(sector_balance)

    with c2:
        st.subheader("Xavf toifalari taqsimoti")
        tier_summary = df["ground_truth_risk_tier"].value_counts()
        st.bar_chart(tier_summary)

    st.markdown("---")
    st.subheader("Hududlar bo'yicha xavf tahlili")
    reg_summary = df.groupby(["region", "ground_truth_risk_tier"]).size().unstack(fill_value=0)
    st.dataframe(reg_summary, use_container_width=True)

# -------------------------------------------------------------
# PAGE 2: TOP-20 WATCH-LIST
# -------------------------------------------------------------
elif menu == "🏆 TOP-20 Yirik Qarz Oluvchilar":
    st.title("🏆 TOP-20 Yirik Qarz Oluvchilar Radari")
    st.markdown("Bank kredit portfelidagi eng yirik 20 ta loyihaning individual nazorati")

    top20_df = df[df["is_top20"]].copy().sort_values(by="loan_balance_uzs", ascending=False)
    top20_balance = top20_df["loan_balance_uzs"].sum()
    share_pct = top20_balance / max(df["loan_balance_uzs"].sum(), 1.0) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("TOP-20 Jami Qoldig'i", f"{top20_balance / 1e12:.2f} trln UZS")
    col2.metric("Umumiy Portfeldagi Ulushi", f"{share_pct:.1f}%")
    col3.metric("Kuzatuvdagi (Sariq/Qizil) Loyihalar", f"{(top20_df['ground_truth_risk_tier'] != RiskTier.GREEN.value).sum()} ta")

    st.markdown("---")

    display_cols = [
        "inn", "company_name", "sector", "region", "loan_balance_uzs",
        "overdue_days", "collateral_coverage_ratio", "revenue_change_pct",
        "ground_truth_risk_tier", "officer_in_charge"
    ]
    st.dataframe(
        top20_df[display_cols].rename(columns={
            "inn": "STIR (INN)",
            "company_name": "Korxona Nomi",
            "sector": "Soha",
            "region": "Hudud",
            "loan_balance_uzs": "Kredit Qoldig'i (UZS)",
            "overdue_days": "Muddati O'tgan Kun",
            "collateral_coverage_ratio": "Garov Qoplami",
            "revenue_change_pct": "Tushum MoM",
            "ground_truth_risk_tier": "Xavf Toifasi",
            "officer_in_charge": "Mas'ul Xodim",
        }),
        use_container_width=True
    )

# -------------------------------------------------------------
# PAGE 3: EARLY WARNING SYSTEM (EWS) ALERTS
# -------------------------------------------------------------
elif menu == "⚠️ Erta Ogohlantirish (EWS)":
    st.title("⚠️ Erta Ogohlantirish Tizimi (EWS) Trigglerlari")
    st.markdown("Regulyator va bank talablariga asosan xatarlarni avtomatik erta aniqlash")

    scan_results = scan_portfolio_ews(df)
    all_alerts = []
    for r in scan_results:
        for a in r.alerts:
            all_alerts.append({
                "STIR (INN)": r.inn,
                "Korxona": r.company_name,
                "TOP-20": "HA" if r.is_top20 else "YO'Q",
                "Qoida ID": a.rule_id,
                "Signal Nomi": a.rule_name,
                "Daraja": a.severity,
                "Aniqlangan Holat": str(a.observed_value),
                "Tavsiya": a.recommended_action,
            })

    alert_columns = [
        "STIR (INN)", "Korxona", "TOP-20", "Qoida ID", "Signal Nomi",
        "Daraja", "Aniqlangan Holat", "Tavsiya"
    ]
    if all_alerts:
        alerts_df = pd.DataFrame(all_alerts)
    else:
        alerts_df = pd.DataFrame(columns=alert_columns)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jami Ogohlantirishlar", f"{len(alerts_df)} ta")
    crit_count = (alerts_df["Daraja"] == "CRITICAL").sum() if "Daraja" in alerts_df.columns else 0
    high_count = (alerts_df["Daraja"] == "HIGH").sum() if "Daraja" in alerts_df.columns else 0
    top_alert_count = (alerts_df["TOP-20"] == "HA").sum() if "TOP-20" in alerts_df.columns else 0

    c2.metric("Kritik Signallar", f"{crit_count} ta")
    c3.metric("Yuqori Xavfli Signallar", f"{high_count} ta")
    c4.metric("TOP-20 Loyihalardagi Signallar", f"{top_alert_count} ta")

    st.markdown("---")

    sev_filter = st.selectbox("Daraja bo'yicha filtrlash:", ["Barchasi", "CRITICAL", "HIGH", "MEDIUM"])
    if sev_filter != "Barchasi" and "Daraja" in alerts_df.columns:
        filtered_alerts = alerts_df[alerts_df["Daraja"] == sev_filter]
    else:
        filtered_alerts = alerts_df

    st.dataframe(filtered_alerts, use_container_width=True)

# -------------------------------------------------------------
# PAGE 4: BORROWER 360 & ML SHAP EXPLAINABILITY
# -------------------------------------------------------------
elif menu == "🔍 Qarz Oluvchi 360° & SHAP":
    st.title("🔍 Qarz Oluvchi 360° Dossier & ML SHAP Tahlili")
    st.markdown("Alohida korxona bo'yicha chuqurlashtirilgan audit, kredit skori va xatar omillari dekompozitsiyasi")

    inn_options = df["inn"].tolist()
    selected_inn = st.selectbox("Korxona STIR (INN) raqamini tanlang yoki kiriting:", inn_options)

    record = df[df["inn"] == selected_inn].iloc[0].to_dict()
    rule_res = ews_engine.evaluate_borrower(record)
    ml_eval = model.evaluate_borrower_record(record)
    explainer = ExplainabilityEngine(model)
    explain_res = explainer.explain_borrower_record(record)

    st.markdown("---")

    # Header Card
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader(record["company_name"])
        st.write(f"**Soha:** {record['sector']} | **Hudud:** {record['region']} | **BXM:** {record['branch_bxm']}")
        st.write(f"**Mas'ul xodim:** {record['officer_in_charge']} | **TOP-20:** {'HA' if record['is_top20'] else 'YO\'Q'}")
    with c2:
        tier = ml_eval["predicted_risk_tier"]
        st.metric("ML Xavf Toifasi", tier)
    with c3:
        st.metric("ML Risk Skori (0-100)", f"{ml_eval['risk_score_0_100']} ball")

    st.markdown("---")

    # Financial & Operational Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kredit Qoldig'i", f"{record['loan_balance_uzs'] / 1e9:.2f} mlrd UZS")
    m2.metric("Garov Qoplami", f"{record['collateral_coverage_ratio']:.2f}x")
    m3.metric("Muddati O'tgan Kun", f"{record['overdue_days']} kun")
    m4.metric("Tushum O'zgarishi MoM", f"{record['revenue_change_pct'] * 100:.1f}%")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("⚠️ EWS Ogohlantirishlari")
        if rule_res.alerts:
            for a in rule_res.alerts:
                st.warning(f"**[{a.rule_id}] {a.rule_name}** ({a.severity})\n\n{a.message}\n\n*Tavsiya:* {a.recommended_action}")
        else:
            st.success("✅ EWS ogohlantirishlari mavjud emas. Ko'rsatkichlar me'yorda.")

    with col_r:
        st.subheader("💡 ML SHAP Risk Omillari Dekompozitsiyasi")
        st.info(f"**Xulosa:** {explain_res['executive_summary_uz']}")

        drivers = explain_res.get("top_risk_drivers", [])
        if drivers:
            st.markdown("**Xatarni oshiruvchi omillar:**")
            for d in drivers:
                st.write(f"- 🔴 **{d['feature_label_uz']}**: +{d['contribution_pts']:.1f} ball ({d['explanation_uz']})")

        protect = explain_res.get("top_protective_factors", [])
        if protect:
            st.markdown("**Xatarni kamaytiruvchi (himoyalovchi) omillar:**")
            for p in protect:
                st.write(f"- 🟢 **{p['feature_label_uz']}**: {p['contribution_pts']:.1f} ball ({p['explanation_uz']})")

    st.markdown("---")

    # Report Generation - Direct single-click download buttons
    st.subheader("📄 Rasmiy Monitoring Dalolatnomasini Yuklab Olish")
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            pdf_path = report_gen.generate_pdf_report(record, rule_res, ml_eval, explain_res, tmp_pdf.name)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        st.download_button(
            label="⬇️ PDF Dalolatnomani Yuklab Olish",
            data=pdf_bytes,
            file_name=f"monitoring_act_{selected_inn}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with btn_col2:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp_md:
            md_path = report_gen.generate_markdown_report(record, rule_res, ml_eval, explain_res, tmp_md.name)
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        st.download_button(
            label="⬇️ Markdown Hisobotni Yuklab Olish",
            data=md_text,
            file_name=f"monitoring_act_{selected_inn}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# -------------------------------------------------------------
# PAGE 5: DOCUMENT AI AUDITOR
# -------------------------------------------------------------
elif menu == "📑 Hujjatlar Auditi (Document AI)":
    st.title("📑 Hujjatlar Auditi (Document AI Inspector)")
    st.markdown(
        "Monitoring dalolatnomalari, filial hisobotlari va audit hujjatlarini "
        "(.docx, .pdf, .xlsx, .txt) sun'iy intellekt yordamida avtomatik tahlil qilish."
    )

    uploaded_file = st.file_uploader(
        "Monitoring hujjatini yuklang:",
        type=["docx", "pdf", "xlsx", "xls", "txt"]
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)

        with st.spinner("Hujjat tahlil qilinmoqda..."):
            audit_res = doc_reviewer.review_file(tmp_path)

        st.success("Tahlil yakunlandi!")
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Hujjat Nomi", audit_res.document_name)
        c2.metric("Tahlil Usuli", audit_res.audit_mode)
        c3.metric("Baholangan Xavf", audit_res.overall_risk_level)

        st.subheader("Xulosa va Statuslar")
        st.write(f"**Umumiy Xulosa:** {audit_res.executive_summary}")
        st.write(f"- **Operatsion holat:** {audit_res.operational_status}")
        st.write(f"- **Garov holati:** {audit_res.collateral_status}")
        st.write(f"- **Moliyaviy salomatlik:** {audit_res.financial_health_status}")

        if audit_res.flags:
            st.subheader(f"Aniqlangan Xavf Belgilari ({len(audit_res.flags)} ta)")
            for flag in audit_res.flags:
                st.error(
                    f"**[{flag.flag_id}] {flag.category}** ({flag.severity})\n\n"
                    f"- *Matndan parcha:* `{flag.context_snippet}`\n"
                    f"- *Tavsiya:* {flag.recommendation}"
                )
        else:
            st.success("Hujjatda shubhali yoki xatarli holatlar aniqlanmadi.")

        st.subheader("Amaliy Harakatlar Rejasi:")
        for act in audit_res.action_plan:
            st.write(f"1. {act}")

# -------------------------------------------------------------
# PAGE 6: MODEL GOVERNANCE & METRICS
# -------------------------------------------------------------
elif menu == "⚙️ ML Boshqaruvi & Metrikalar":
    st.title("⚙️ Model Boshqaruvi & Validatsiya Metriklari")
    st.markdown("Markaziy Bank va Model Risk Governance talablari asosida model sifatini baholash")

    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()
    eval_metrics = evaluate_model_performance(model, X, y)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Umumiy Aniqlik (Accuracy)", f"{eval_metrics['accuracy'] * 100:.2f}%")
    m2.metric("EWS Precision (Kuzatuv)", f"{eval_metrics['ews_watch_precision'] * 100:.2f}%", "Target: >=70%")
    m3.metric("EWS Recall (Kuzatuv)", f"{eval_metrics['ews_watch_recall'] * 100:.2f}%", "Target: >=65%")
    m4.metric("ROC-AUC (OVR)", f"{eval_metrics['roc_auc_ovr']:.4f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Klasslar bo'yicha metrikalar")
        st.dataframe(pd.DataFrame(eval_metrics["per_class_metrics"]).T, use_container_width=True)

    with c2:
        st.subheader("Xatolik matritsasi (Confusion Matrix)")
        cm_data = eval_metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm_data["matrix"],
            index=[f"Haqiqiy {l}" for l in cm_data["labels"]],
            columns=[f"Bashorat {l}" for l in cm_data["labels"]],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Eng Muhim Xususiyatlar (Global Feature Importances)")
    explainer = ExplainabilityEngine(model)
    imp_df = explainer.get_global_feature_importances(top_n=12)
    st.bar_chart(imp_df.set_index("label_uz")["importance"])
