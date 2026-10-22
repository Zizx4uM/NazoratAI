"""
Automated Report Generator for NazoratAI.
Produces:
1. Executive Markdown audit acts and portfolio summaries.
2. Formatted PDF Credit Monitoring Acts with ML & SHAP explainability using ReportLab.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import pandas as pd

from nazorat_ai.config import RiskTier, EWSSeverity
from nazorat_ai.ews.rules import RuleResult

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def _sanitize_pdf_text(text: str) -> str:
    """Normalize special typographic characters to standard Latin-1 compatible chars for ReportLab."""
    if not text:
        return ""
    s = str(text)
    # Replace Uzbek modifier letters & curly quotes
    s = s.replace("\u02bb", "'").replace("\u02bc", "'").replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "--").replace("\u2013", "-")
    return s


class ReportGenerator:
    """Generates official credit monitoring reports in Markdown and PDF."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        record: Dict,
        rule_result: RuleResult,
        ml_score: Dict,
        explain_dict: Optional[Dict] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Generate detailed Markdown Credit Monitoring Act for a single corporate borrower.
        """
        inn = str(record.get("inn", "000000000"))
        company = str(record.get("company_name", "Noma'lum Korxona"))
        sector = str(record.get("sector", "Noma'lum"))
        region = str(record.get("region", "Noma'lum"))
        bxm = str(record.get("branch_bxm", "Bosh ofis"))
        officer = str(record.get("officer_in_charge", "Noma'lum"))
        is_top20 = "HA (TOP-20)" if record.get("is_top20") else "YO'Q"
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        tier = rule_result.risk_tier
        tier_badges = {
            RiskTier.GREEN.value: "🟢 YASHIL (STANDART)",
            RiskTier.YELLOW.value: "🟡 SARIQ (POTENSIAL MUAMMOLI / KUZATUV)",
            RiskTier.RED.value: "🔴 QIZIL (SUBSTANDART / SUBXALI)",
        }
        badge = tier_badges.get(tier, tier)

        lines = [
            f"# KREDIT MONITORINGI VA XATARLARNI ERTA ANIQLASH (EWS) DALOLATNOMASI",
            f"**Hujjat raqami:** KM-ACT-{inn}-{datetime.now().strftime('%Y%m%d')}",
            f"**Shakllantirilgan sana va vaqt:** {gen_time}",
            f"**Tizim:** NazoratAI Automated Credit Intelligence",
            "",
            "---",
            "",
            "## 1. QARZ OLUVCHI REKVIZITLARI VA UMUMIY MA'LUMOT",
            "",
            "| Ko'rsatkich | Qiymat |",
            "| :--- | :--- |",
            f"| **Korxona nomi** | {company} |",
            f"| **STIR (INN)** | `{inn}` |",
            f"| **Faoliyat sohasi** | {sector} |",
            f"| **Hudud (Viloyat)** | {region} |",
            f"| **Xizmat ko'rsatuvchi BXM** | {bxm} |",
            f"| **Mas'ul monitoring xodimi** | {officer} |",
            f"| **Yirik qarz oluvchi (TOP-20)** | **{is_top20}** |",
            f"| **Monitoring davriyligi** | {record.get('monitoring_cycle', 'Har chorakda')} |",
            f"| **Oxirgi monitoring sanasi** | {record.get('last_monitoring_date', 'N/A')} |",
            "",
            "---",
            "",
            "## 2. KREDIT SHARTLARI VA MOLIYAVIY KO'RSATKICHLAR",
            "",
            "| Moliyaviy / Kredit parametri | Joriy qiymat | Me'yor / Ta'rif |",
            "| :--- | :--- | :--- |",
            f"| Ajratilgan kredit summasi | {record.get('loan_amount_uzs', 0):,.2f} UZS | Asosiy qarz shartnomasi |",
            f"| Joriy kredit qoldig'i | {record.get('loan_balance_uzs', 0):,.2f} UZS | To'lanishi lozim qoldiq |",
            f"| Garov baholash qiymati | {record.get('collateral_value_uzs', 0):,.2f} UZS | Rasmiy baholash dalolatnomasi |",
            f"| Garov qoplami koeffitsienti | **{record.get('collateral_coverage_ratio', 0):.2f}x** | Minimal talab >= 1.0x |",
            f"| Muddati o'tgan kunlar | **{record.get('overdue_days', 0)} kun** | >45 kun (Sariq), >90 kun (Qizil) |",
            f"| 2-Kartoteka qarzdorligi | {record.get('k2_card_index_amount_uzs', 0):,.2f} UZS | Talabnoma mavjudligi |",
            f"| Oylik tushum o'zgarishi (MoM) | **{record.get('revenue_change_pct', 0) * 100:.1f}%** | Pasayish xavfi >= 15% |",
            f"| Ishlab chiqarish hajmi (MoM) | **{record.get('production_change_pct', 0) * 100:.1f}%** | Pasayish xavfi >= 20% |",
            f"| Elektr energiyasi sarfi (MoM) | {record.get('electricity_change_pct', 0) * 100:.1f}% | Ishlab turganlik indikatori |",
            f"| DSCR (Qarzni qoplash koeffitsienti) | {record.get('dscr', 0):.2f} | Tavsiya etiladi >= 1.10 |",
            f"| KATM kredit reyting skori | {record.get('katm_credit_score', 0)} / 500 | Kredit byurosi ma'lumoti |",
            "",
            "---",
            "",
            f"## 3. ERTA ANIQLASH TIZIMI (EWS) SIGNALLARI VA XATARLAR",
            f"**Aniqlangan jami signallar:** {rule_result.total_alerts} ta | "
            f"**Kritik signallar:** {rule_result.critical_alerts_count} ta | "
            f"**Yakuniy xavf toifasi:** {badge}",
            "",
        ]

        if rule_result.alerts:
            lines.extend([
                "| Qoida ID | Signal nomi | Darajasi | Aniqlangan holat | Tavsiya etilgan chora |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for a in rule_result.alerts:
                lines.append(
                    f"| **{a.rule_id}** | {a.rule_name} | `{a.severity}` | {a.observed_value} ({a.threshold_value}) | {a.recommended_action} |"
                )
        else:
            lines.append("✅ *Hech qanday ogohlantiruvchi signal aniqlanmadi. Korxona parametrlari me'yor doirasida.*")

        lines.extend([
            "",
            "---",
            "",
            "## 4. SUN'IY INTELLEKT (ML) SKORINGI VA SHAP TUSHUNTIRISHLARI",
            "",
            f"- **Bashorat qilingan risk toifasi:** **{ml_score.get('predicted_risk_tier')}**",
            f"- **Uzluksiz risk skori:** **{ml_score.get('risk_score_0_100', 0)} / 100 ball** (0: eng xavfsiz, 100: eng yuqori risk)",
            f"- **Ehtimolliklar taqsimoti:** Yashil: {ml_score.get('prob_green', 0):.1%} | "
            f"Sariq: {ml_score.get('prob_yellow', 0):.1%} | "
            f"Qizil: {ml_score.get('prob_red', 0):.1%}",
            "",
        ])

        if explain_dict:
            lines.extend([
                f"### 4.1. Asosiy xatar omillari (SHAP / Feature Attribution):",
                f"> **Xulosa:** {explain_dict.get('executive_summary_uz', '')}",
                "",
                "| Omil nomi | Qiymati | Riskka qo'shgan hissasi | Ta'sir yo'nalishi |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for d in explain_dict.get("top_risk_drivers", []):
                lines.append(
                    f"| {d['feature_label_uz']} | {d['feature_value']:,.2f} | **+{d['contribution_pts']:.1f} ball** | ⚠️ Xatarni oshiruvchi |"
                )
            for p in explain_dict.get("top_protective_factors", []):
                lines.append(
                    f"| {p['feature_label_uz']} | {p['feature_value']:,.2f} | **{p['contribution_pts']:.1f} ball** | 🛡️ Xatarni kamaytiruvchi |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## 5. XULOSA VA AMALIY TAVSIYALAR",
            "",
            f"**Tizim xulosasi:** {rule_result.recommended_action_summary}",
            "",
            "### Amalga oshirilishi shart bo'lgan harakatlar:",
            "1. **BXM Mas'ul xodimi:** Mazkur dalolatnoma asosida 3 ish kuni ichida mijoz faoliyatini joyida tekshirish.",
            "2. **Garov ta'minoti:** Garov mulkining butunligi va amaldagi mavjudligini qayta ko'zdan kechirish.",
            "3. **Tushumlar nazorati:** Mijozning tushumlarini boshqa bank hisobraqamlariga o'tganligini soliq organlari orqali solishtirish.",
            "",
            "",
            "**Monitoring xodimi imzosi:** _____________________ / " + officer + " /",
            "",
            "**BXM Boshqaruvchisi imzosi:** _____________________ / _________________ /",
        ])

        content = "\n".join(lines)
        if output_file is None:
            output_file = self.output_dir / f"monitoring_act_{inn}_{datetime.now().strftime('%Y%m%d')}.md"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        return output_file

    def generate_pdf_report(
        self,
        record: Dict,
        rule_result: RuleResult,
        ml_score: Dict,
        explain_dict: Optional[Dict] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Generate official formatted PDF Credit Monitoring Act using ReportLab.
        Includes complete ML explainability & SHAP driver table.
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab is required for PDF generation.")

        inn = _sanitize_pdf_text(str(record.get("inn", "000000000")))
        company = _sanitize_pdf_text(str(record.get("company_name", "Noma'lum Korxona")))
        if output_file is None:
            output_file = self.output_dir / f"monitoring_act_{inn}_{datetime.now().strftime('%Y%m%d')}.pdf"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_file),
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#0d47a1"),
            spaceAfter=4,
        )
        h2_style = ParagraphStyle(
            "DocH2",
            parent=styles["Heading2"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1565c0"),
            spaceBefore=6,
            spaceAfter=3,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
        )
        bold_style = ParagraphStyle(
            "DocBold",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            fontName="Helvetica-Bold",
        )

        story = []

        # Title & Header
        story.append(Paragraph("<b>NAZORAT AI — KREDIT MONITORINGI VA EWS DALOLATNOMASI</b>", title_style))
        story.append(Paragraph(
            f"<b>Hujjat:</b> KM-{inn} | <b>Sana:</b> {datetime.now().strftime('%d.%m.%Y %H:%M')} | "
            f"<b>Tizim:</b> Avtomatlashtirilgan kredit nazorati",
            body_style
        ))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d47a1"), spaceAfter=6))

        # Risk Tier Banner
        tier = rule_result.risk_tier
        if tier == RiskTier.RED.value:
            bg_col = colors.HexColor("#ffebee")
            txt_col = colors.HexColor("#c62828")
            tier_uz = "QIZIL TOIFA (SUBSTANDART / KRITIK XATAR)"
        elif tier == RiskTier.YELLOW.value:
            bg_col = colors.HexColor("#fffde7")
            txt_col = colors.HexColor("#f57f17")
            tier_uz = "SARIQ TOIFA (POTENSIAL MUAMMOLI / KUZATUV RO'YXATI)"
        else:
            bg_col = colors.HexColor("#e8f5e9")
            txt_col = colors.HexColor("#2e7d32")
            tier_uz = "YASHIL TOIFA (STANDART KREDIT)"

        banner_text = f"<b>STATUS: {tier_uz}</b> | ML Xatar Skori: <b>{ml_score.get('risk_score_0_100', 0):.1f} / 100</b>"
        banner_table = Table([[Paragraph(banner_text, ParagraphStyle("B", parent=body_style, textColor=txt_col, alignment=1))]], colWidths=[520])
        banner_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_col),
            ("BOX", (0, 0), (-1, -1), 1.0, txt_col),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 6))

        # 1. Company Profile Table
        story.append(Paragraph("<b>1. Qarz Oluvchi Rekvizitlari</b>", h2_style))
        profile_data = [
            [Paragraph("<b>Korxona nomi</b>", body_style), Paragraph(company, body_style),
             Paragraph("<b>STIR (INN)</b>", body_style), Paragraph(inn, body_style)],
            [Paragraph("<b>Faoliyat sohasi</b>", body_style), Paragraph(_sanitize_pdf_text(str(record.get("sector", ""))), body_style),
             Paragraph("<b>Hudud</b>", body_style), Paragraph(_sanitize_pdf_text(str(record.get("region", ""))), body_style)],
            [Paragraph("<b>Xizmat BXM</b>", body_style), Paragraph(_sanitize_pdf_text(str(record.get("branch_bxm", ""))), body_style),
             Paragraph("<b>Mas'ul xodim</b>", body_style), Paragraph(_sanitize_pdf_text(str(record.get("officer_in_charge", ""))), body_style)],
            [Paragraph("<b>TOP-20 Qarz oluvchi</b>", body_style), Paragraph("HA" if record.get("is_top20") else "YO'Q", bold_style),
             Paragraph("<b>Davriylik</b>", body_style), Paragraph(_sanitize_pdf_text(str(record.get("monitoring_cycle", "Har chorakda"))), body_style)],
        ]
        t_profile = Table(profile_data, colWidths=[110, 150, 110, 150])
        t_profile.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_profile)
        story.append(Spacer(1, 6))

        # 2. Key Financial Indicators
        story.append(Paragraph("<b>2. Asosiy Kredit va Moliyaviy Ko'rsatkichlar</b>", h2_style))
        fin_data = [
            [Paragraph("<b>Ko'rsatkich</b>", bold_style), Paragraph("<b>Qiymat</b>", bold_style),
             Paragraph("<b>Ko'rsatkich</b>", bold_style), Paragraph("<b>Qiymat</b>", bold_style)],
            [Paragraph("Kredit miqdori", body_style), Paragraph(f"{record.get('loan_amount_uzs', 0):,.0f} UZS", body_style),
             Paragraph("Kredit qoldig'i", body_style), Paragraph(f"{record.get('loan_balance_uzs', 0):,.0f} UZS", body_style)],
            [Paragraph("Garov bahosi", body_style), Paragraph(f"{record.get('collateral_value_uzs', 0):,.0f} UZS", body_style),
             Paragraph("Garov qoplami", body_style), Paragraph(f"<b>{record.get('collateral_coverage_ratio', 0):.2f}x</b>", body_style)],
            [Paragraph("Muddati o'tgan kun", body_style), Paragraph(f"<b>{record.get('overdue_days', 0)} kun</b>", body_style),
             Paragraph("2-Kartoteka", body_style), Paragraph(f"{record.get('k2_card_index_amount_uzs', 0):,.0f} UZS", body_style)],
            [Paragraph("Tushum MoM", body_style), Paragraph(f"{record.get('revenue_change_pct', 0) * 100:.1f}%", body_style),
             Paragraph("Ishlab chiqarish MoM", body_style), Paragraph(f"{record.get('production_change_pct', 0) * 100:.1f}%", body_style)],
            [Paragraph("DSCR", body_style), Paragraph(f"{record.get('dscr', 0):.2f}", body_style),
             Paragraph("KATM Reyting Skori", body_style), Paragraph(f"{record.get('katm_credit_score', 0)}", body_style)],
        ]
        t_fin = Table(fin_data, colWidths=[130, 130, 130, 130])
        t_fin.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3f2fd")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbdefb")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_fin)
        story.append(Spacer(1, 6))

        # 3. EWS Alerts Table
        story.append(Paragraph(f"<b>3. Erta Ogohlantirish Tizimi (EWS) Signallari ({rule_result.total_alerts} ta)</b>", h2_style))
        if rule_result.alerts:
            alert_rows = [[
                Paragraph("<b>ID</b>", bold_style),
                Paragraph("<b>Xatar Nomi</b>", bold_style),
                Paragraph("<b>Daraja</b>", bold_style),
                Paragraph("<b>Aniqlangan Qiymat</b>", bold_style),
                Paragraph("<b>Tavsiya Etilgan Harakat</b>", bold_style),
            ]]
            for a in rule_result.alerts:
                alert_rows.append([
                    Paragraph(_sanitize_pdf_text(a.rule_id), body_style),
                    Paragraph(_sanitize_pdf_text(a.rule_name), body_style),
                    Paragraph(_sanitize_pdf_text(a.severity), bold_style),
                    Paragraph(_sanitize_pdf_text(str(a.observed_value)), body_style),
                    Paragraph(_sanitize_pdf_text(a.recommended_action), body_style),
                ])
            t_alerts = Table(alert_rows, colWidths=[50, 150, 55, 95, 170])
            t_alerts.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ffe0b2")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffcc80")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(t_alerts)
        else:
            story.append(Paragraph("<i>Ogohlantiruvchi signallar yo'q. Parametrlar me'yorda.</i>", body_style))
        story.append(Spacer(1, 6))

        # 4. ML Risk Scoring & SHAP Explainability (RENDERED CLEANLY)
        story.append(Paragraph("<b>4. Sun'iy Intellekt (ML) Skoringi va SHAP Tushuntirishlari</b>", h2_style))
        ml_summary_text = (
            f"<b>Bashorat:</b> {ml_score.get('predicted_risk_tier')} | "
            f"<b>Risk Skori:</b> {ml_score.get('risk_score_0_100', 0):.1f} / 100 | "
            f"<b>Ehtimolliklar:</b> Yashil {ml_score.get('prob_green', 0):.1%}, "
            f"Sariq {ml_score.get('prob_yellow', 0):.1%}, "
            f"Qizil {ml_score.get('prob_red', 0):.1%}"
        )
        story.append(Paragraph(ml_summary_text, body_style))

        if explain_dict and (explain_dict.get("top_risk_drivers") or explain_dict.get("top_protective_factors")):
            shap_rows = [[
                Paragraph("<b>Omil Nomi (Feature)</b>", bold_style),
                Paragraph("<b>Qiymat</b>", bold_style),
                Paragraph("<b>Qo'shgan Hissasi</b>", bold_style),
                Paragraph("<b>Ta'sir Yo'nalishi</b>", bold_style),
            ]]
            for d in explain_dict.get("top_risk_drivers", [])[:4]:
                shap_rows.append([
                    Paragraph(_sanitize_pdf_text(d["feature_label_uz"]), body_style),
                    Paragraph(f"{d['feature_value']:,.2f}", body_style),
                    Paragraph(f"+{d['contribution_pts']:.1f} ball", bold_style),
                    Paragraph("Xatarni oshiruvchi", body_style),
                ])
            for p in explain_dict.get("top_protective_factors", [])[:3]:
                shap_rows.append([
                    Paragraph(_sanitize_pdf_text(p["feature_label_uz"]), body_style),
                    Paragraph(f"{p['feature_value']:,.2f}", body_style),
                    Paragraph(f"{p['contribution_pts']:.1f} ball", bold_style),
                    Paragraph("Himoyalovchi (pasaytiruvchi)", body_style),
                ])
            t_shap = Table(shap_rows, colWidths=[200, 100, 100, 120])
            t_shap.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede7f6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1c4e9")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(Spacer(1, 4))
            story.append(t_shap)

        story.append(Spacer(1, 6))

        # 5. Action and Signatures
        story.append(Paragraph("<b>5. Xulosa va Imzolar</b>", h2_style))
        story.append(Paragraph(f"<b>Xulosa:</b> {_sanitize_pdf_text(rule_result.recommended_action_summary)}", body_style))
        story.append(Spacer(1, 10))

        sig_data = [
            [Paragraph("<b>Monitoring xodimi:</b> ____________________", body_style),
             Paragraph("<b>BXM Boshqaruvchisi:</b> ____________________", body_style)],
            [Paragraph(f"F.I.SH: {_sanitize_pdf_text(str(record.get('officer_in_charge', '')))}", body_style),
             Paragraph("F.I.SH: _________________________________", body_style)],
        ]
        t_sig = Table(sig_data, colWidths=[260, 260])
        t_sig.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_sig)

        doc.build(story)
        return output_file


def generate_borrower_markdown_report(
    record: Dict,
    rule_result: RuleResult,
    ml_score: Dict,
    explain_dict: Optional[Dict] = None,
    output_file: Optional[Union[str, Path]] = None,
) -> Path:
    gen = ReportGenerator()
    return gen.generate_markdown_report(record, rule_result, ml_score, explain_dict, output_file)


def generate_borrower_pdf_report(
    record: Dict,
    rule_result: RuleResult,
    ml_score: Dict,
    explain_dict: Optional[Dict] = None,
    output_file: Optional[Union[str, Path]] = None,
) -> Path:
    gen = ReportGenerator()
    return gen.generate_pdf_report(record, rule_result, ml_score, explain_dict, output_file)


def generate_portfolio_summary_report(
    df: pd.DataFrame,
    scan_results: List[RuleResult],
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generate aggregate executive portfolio health summary report for Board & Central Bank.
    Guarded against empty DataFrames.
    """
    out_dir = Path(output_dir) if output_dir else Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"portfolio_executive_summary_{datetime.now().strftime('%Y%m%d')}.md"

    total_borrowers = max(len(df), 1)
    actual_count = len(df)
    total_loans = df["loan_amount_uzs"].sum() if "loan_amount_uzs" in df.columns else 0.0
    total_balance = df["loan_balance_uzs"].sum() if "loan_balance_uzs" in df.columns else 0.0

    green_count = sum(1 for r in scan_results if r.risk_tier == RiskTier.GREEN.value)
    yellow_count = sum(1 for r in scan_results if r.risk_tier == RiskTier.YELLOW.value)
    red_count = sum(1 for r in scan_results if r.risk_tier == RiskTier.RED.value)

    top20_df = df[df["is_top20"]] if "is_top20" in df.columns else df.head(20)
    top20_balance = top20_df["loan_balance_uzs"].sum() if "loan_balance_uzs" in top20_df.columns else 0.0

    content = f"""# NAZORAT AI: KREDIT PORTFELI VA EWS IJRO XULOSASI
**Hujjat:** Boshqaruv va Markaziy Bank uchun choraklik tahliliy hisobot
**Sana:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Qamrov:** {actual_count} ta korporativ qarz oluvchilar

---

## 1. Portfelning Umumiy Holati
- **Jami qarz oluvchilar soni:** {actual_count} ta
- **Jami kredit shartnomalari hajmi:** {total_loans:,.0f} UZS
- **Joriy kredit qoldig'i (Exposure):** {total_balance:,.0f} UZS
- **TOP-20 yirik qarz oluvchilar qoldig'i:** {top20_balance:,.0f} UZS ({top20_balance / max(total_balance, 1.0) * 100:.1f}% ulush)

---

## 2. Xavf Toifalari Taqsimoti (EWS Klassifikatsiyasi)
- 🟢 **Yashil toifa (Standart):** {green_count} ta ({green_count / total_borrowers * 100:.1f}%) — har 6 oyda monitoring
- 🟡 **Sariq toifa (Kuzatuv / Potensial muammoli):** {yellow_count} ta ({yellow_count / total_borrowers * 100:.1f}%) — favqulodda monitoring
- 🔴 **Qizil toifa (Substandart / NPL):** {red_count} ta ({red_count / total_borrowers * 100:.1f}%) — sud va garov monitoringi

---

## 3. Asosiy EWS Trigglerlari Bo'yicha Statistika
- Muddati o'tgan qarzdorlik > 45 kun: {sum(1 for _, row in df.iterrows() if row.get('overdue_days', 0) > 45)} ta korxona
- Muddati o'tgan qarzdorlik > 90 kun (NPL): {sum(1 for _, row in df.iterrows() if row.get('overdue_days', 0) > 90)} ta korxona
- Oylik tushumi >= 15% ga pasaygan: {sum(1 for _, row in df.iterrows() if row.get('revenue_change_pct', 0) <= -0.15)} ta korxona
- Ishlab chiqarish hajmi >= 20% ga kamaygan: {sum(1 for _, row in df.iterrows() if row.get('production_change_pct', 0) <= -0.20)} ta korxona
- 2-Kartoteka qarzdorligi mavjud: {sum(1 for _, row in df.iterrows() if row.get('k2_card_index_amount_uzs', 0) > 0)} ta korxona
- Garov ta'minoti qoplami < 100%: {sum(1 for _, row in df.iterrows() if row.get('collateral_coverage_ratio', 1.0) < 1.0)} ta korxona

---
*Ushbu hisobot NazoratAI kredit monitoring tizimi tomonidan avtomatik generatsiya qilindi.*
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)

    return report_file
