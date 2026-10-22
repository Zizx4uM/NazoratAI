"""
Early Warning System (EWS) Rules Engine for Banking Corporate Credit Monitoring.
Evaluates regulatory and operational trigger conditions mandated by
the Central Bank of Uzbekistan and commercial credit monitoring standards.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd

from nazorat_ai.config import (
    OVERDUE_DAYS_YELLOW_THRESHOLD,
    OVERDUE_DAYS_RED_THRESHOLD,
    REVENUE_DROP_MOM_THRESHOLD,
    PRODUCTION_DROP_MOM_THRESHOLD,
    ELECTRICITY_DROP_MOM_THRESHOLD,
    COLLATERAL_COVERAGE_THRESHOLD,
    DSCR_THRESHOLD,
    K2_CARD_INDEX_THRESHOLD_UZS,
    EWSSeverity,
    RiskTier,
)


@dataclass
class EWSAlert:
    """Structure representing a single early warning alert."""
    rule_id: str
    rule_name: str
    severity: str
    trigger_condition: str
    observed_value: Any
    threshold_value: Any
    recommended_action: str
    message: str


@dataclass
class RuleResult:
    """Overall result of evaluating EWS rules for a single borrower."""
    inn: str
    company_name: str
    is_top20: bool
    risk_tier: str
    total_alerts: int
    critical_alerts_count: int
    high_alerts_count: int
    alerts: List[EWSAlert]
    recommended_action_summary: str


class EWSEngine:
    """Evaluates rule-based credit early warning signals."""

    def __init__(self):
        pass

    def evaluate_borrower(self, row: Dict[str, Any]) -> RuleResult:
        """
        Evaluate all EWS rules against a single borrower dictionary / record.
        """
        inn = str(row.get("inn", "UNKNOWN"))
        company_name = str(row.get("company_name", "Noma'lum korxona"))
        is_top20 = bool(row.get("is_top20", False))
        alerts: List[EWSAlert] = []

        # 1. Overdue > 90 days (RED default trigger)
        overdue_days = int(row.get("overdue_days", 0))
        if overdue_days > OVERDUE_DAYS_RED_THRESHOLD:
            alerts.append(EWSAlert(
                rule_id="EWS-01",
                rule_name="Muddati o'tgan qarzdorlik > 90 kun (NPL / Qizil toifa)",
                severity=EWSSeverity.CRITICAL.value,
                trigger_condition=f"overdue_days > {OVERDUE_DAYS_RED_THRESHOLD}",
                observed_value=f"{overdue_days} kun",
                threshold_value=f"> {OVERDUE_DAYS_RED_THRESHOLD} kun",
                recommended_action="Zudlik bilan sudga da'vo arizasi kiritish va garov mulkini xatlash.",
                message=f"Kredit to'lovi muddati {overdue_days} kunga kechikkan. Kredit 'Qizil' toifaga o'tkaziladi.",
            ))
        # 2. Overdue > 45 days (YELLOW emergency monitoring trigger)
        elif overdue_days > OVERDUE_DAYS_YELLOW_THRESHOLD:
            alerts.append(EWSAlert(
                rule_id="EWS-02",
                rule_name="Muddati o'tgan qarzdorlik > 45 kun (Sariq toifa)",
                severity=EWSSeverity.CRITICAL.value,
                trigger_condition=f"overdue_days > {OVERDUE_DAYS_YELLOW_THRESHOLD}",
                observed_value=f"{overdue_days} kun",
                threshold_value=f"> {OVERDUE_DAYS_YELLOW_THRESHOLD} kun",
                recommended_action="Favqulodda monitoring dalolatnomasi rasmiylashtirish va joyiga chiqish.",
                message=f"Kredit to'lovi muddati {overdue_days} kunga kechikkan (Sariq toifa). Favqulodda monitoring talab etiladi.",
            ))

        # 3. Revenue slump MoM >= 15%
        revenue_change = float(row.get("revenue_change_pct", 0.0))
        if revenue_change <= REVENUE_DROP_MOM_THRESHOLD:
            drop_pct = abs(revenue_change) * 100
            alerts.append(EWSAlert(
                rule_id="EWS-03",
                rule_name="Pul tushumi pasayishi (>= 15% MoM)",
                severity=EWSSeverity.HIGH.value,
                trigger_condition=f"revenue_change_pct <= {REVENUE_DROP_MOM_THRESHOLD}",
                observed_value=f"-{drop_pct:.1f}%",
                threshold_value="<= -15.0%",
                recommended_action="Mijozning soliq va bank hisobraqamlari aylanmasini tahlil qilish.",
                message=f"Korxona oylik pul tushumi {drop_pct:.1f}% ga pasaygan. Pul oqimi xavfi mavjud.",
            ))

        # 4. Production volume drop MoM >= 20%
        prod_change = float(row.get("production_change_pct", 0.0))
        if prod_change <= PRODUCTION_DROP_MOM_THRESHOLD:
            drop_pct = abs(prod_change) * 100
            alerts.append(EWSAlert(
                rule_id="EWS-04",
                rule_name="Ishlab chiqarish hajmi pasayishi (>= 20% MoM)",
                severity=EWSSeverity.HIGH.value,
                trigger_condition=f"production_change_pct <= {PRODUCTION_DROP_MOM_THRESHOLD}",
                observed_value=f"-{drop_pct:.1f}%",
                threshold_value="<= -20.0%",
                recommended_action="Korxona ishlab chiqarish quvvatlari yuklamasini joyiga chiqib tekshirish.",
                message=f"Ishlab chiqarish hajmi o'tgan davrga nisbatan {drop_pct:.1f}% ga kamaygan.",
            ))

        # 5. K-2 Card index 2 debt > 0
        k2_amount = float(row.get("k2_card_index_amount_uzs", 0.0))
        if k2_amount > K2_CARD_INDEX_THRESHOLD_UZS:
            alerts.append(EWSAlert(
                rule_id="EWS-05",
                rule_name="2-Kartoteka qarzdorligi mavjudligi",
                severity=EWSSeverity.CRITICAL.value if k2_amount > 100e6 else EWSSeverity.HIGH.value,
                trigger_condition="k2_card_index_amount_uzs > 0",
                observed_value=f"{k2_amount:,.0f} UZS",
                threshold_value="0 UZS",
                recommended_action="2-kartotekadagi to'lov talabnomalarining sabablarini o'rganish.",
                message=f"Mijoz hisobida {k2_amount:,.0f} so'm 2-kartoteka qarzdorligi qayd etilgan.",
            ))

        # 6. Electricity consumption drop >= 25% (Operational activity indicator)
        elec_change = float(row.get("electricity_change_pct", 0.0))
        if elec_change <= ELECTRICITY_DROP_MOM_THRESHOLD:
            drop_pct = abs(elec_change) * 100
            alerts.append(EWSAlert(
                rule_id="EWS-06",
                rule_name="Elektr energiya iste'moli pasayishi (>= 25% MoM)",
                severity=EWSSeverity.MEDIUM.value,
                trigger_condition=f"electricity_change_pct <= {ELECTRICITY_DROP_MOM_THRESHOLD}",
                observed_value=f"-{drop_pct:.1f}%",
                threshold_value="<= -25.0%",
                recommended_action="Zavod/sex faoliyati to'xtaganligini BXM xodimlari orqali aniqlash.",
                message=f"Elektr energiyasi iste'moli {drop_pct:.1f}% ga pasaygan (ish to'xtashi alomati).",
            ))

        # 7. Collateral Coverage Ratio < 1.0 (Under-collateralized)
        collateral_ratio = float(row.get("collateral_coverage_ratio", 1.5))
        if collateral_ratio < COLLATERAL_COVERAGE_THRESHOLD:
            alerts.append(EWSAlert(
                rule_id="EWS-07",
                rule_name="Garov ta'minoti yetishmovchiligi (< 100%)",
                severity=EWSSeverity.HIGH.value,
                trigger_condition=f"collateral_coverage_ratio < {COLLATERAL_COVERAGE_THRESHOLD}",
                observed_value=f"{collateral_ratio * 100:.1f}%",
                threshold_value="< 100.0%",
                recommended_action="Garov mulkini qayta baholash va qo'shimcha garov ta'minoti jalb qilish.",
                message=f"Garov qoplami qarz qoldig'iga nisbatan {collateral_ratio * 100:.1f}% ni tashkil etmoqda.",
            ))

        # 8. DSCR < 1.10 (Debt Service Coverage)
        dscr = float(row.get("dscr", 1.5))
        if dscr < DSCR_THRESHOLD:
            alerts.append(EWSAlert(
                rule_id="EWS-08",
                rule_name="Qarzga xizmat ko'rsatish koeffitsienti past (DSCR < 1.10)",
                severity=EWSSeverity.MEDIUM.value,
                trigger_condition=f"dscr < {DSCR_THRESHOLD}",
                observed_value=f"{dscr:.2f}",
                threshold_value=f"< {DSCR_THRESHOLD:.2f}",
                recommended_action="Moliyaviy oqimlarni qayta ko'rib chiqish va to'lov grafigini tahlil qilish.",
                message=f"DSCR {dscr:.2f} bo'lib, oylik operatsion foyda kredit qarzini qoplashga zo'rg'a yetmoqda.",
            ))

        # 9. TOP-20 special watch trigger
        if is_top20 and len(alerts) > 0:
            alerts.append(EWSAlert(
                rule_id="EWS-09",
                rule_name="Yirik qarz oluvchi (TOP-20) xatar signali",
                severity=EWSSeverity.HIGH.value,
                trigger_condition="is_top20 == True and total_alerts > 0",
                observed_value="TOP-20 Korxona",
                threshold_value="Maxsus nazorat",
                recommended_action="Departament boshlig'i va Boshqaruv a'zolariga shoshilinch axborot kiritish.",
                message="Bankning eng yirik 20 qarz oluvchilaridan birida salbiy dinamika aniqlandi.",
            ))

        # Determine implied risk tier from rule fires
        critical_count = sum(1 for a in alerts if a.severity == EWSSeverity.CRITICAL.value)
        high_count = sum(1 for a in alerts if a.severity == EWSSeverity.HIGH.value)

        if overdue_days > OVERDUE_DAYS_RED_THRESHOLD or critical_count >= 2:
            computed_tier = RiskTier.RED.value
            action_summary = "Qizil toifa: Sud jarayonini tayyorlash va garovni xatlash."
        elif overdue_days > OVERDUE_DAYS_YELLOW_THRESHOLD or critical_count >= 1 or high_count >= 2:
            computed_tier = RiskTier.YELLOW.value
            action_summary = "Sariq toifa: Favqulodda monitoring dalolatnomasi rasmiylashtirish."
        elif len(alerts) > 0:
            computed_tier = RiskTier.YELLOW.value
            action_summary = "Nazorat ostida: Keyingi choraklik monitoringda chuqurlashtirilgan tekshirish."
        else:
            computed_tier = RiskTier.GREEN.value
            action_summary = "Yashil toifa: Standart monitoring (har 6 oyda bir marta)."

        return RuleResult(
            inn=inn,
            company_name=company_name,
            is_top20=is_top20,
            risk_tier=computed_tier,
            total_alerts=len(alerts),
            critical_alerts_count=critical_count,
            high_alerts_count=high_count,
            alerts=alerts,
            recommended_action_summary=action_summary,
        )


def evaluate_borrower_ews(row: Dict[str, Any]) -> RuleResult:
    """Convenience function to evaluate a single borrower record."""
    engine = EWSEngine()
    return engine.evaluate_borrower(row)


def scan_portfolio_ews(df: pd.DataFrame) -> List[RuleResult]:
    """
    Scan entire portfolio DataFrame and return list of RuleResults.
    """
    engine = EWSEngine()
    results: List[RuleResult] = []
    for _, row in df.iterrows():
        results.append(engine.evaluate_borrower(row.to_dict()))
    return results
