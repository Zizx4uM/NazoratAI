"""
Realistic synthetic enterprise credit dataset generator for NazoratAI.
Models 603 corporate borrowers in Uzbekistan across 10 key sectors and 14 regions,
including the TOP-20 largest credit exposures and Central Bank regulatory tiers.
"""

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd

from nazorat_ai.config import (
    PORTFOLIO_TOTAL_BORROWERS,
    PORTFOLIO_TOP20_COUNT,
    SECTORS,
    REGIONS,
    RiskTier,
)

# Realistic enterprise naming components
COMPANY_PREFIXES = [
    "O'zbekiston", "Toshkent", "Samarqand", "Buxoro", "Farg'ona", "Andijon",
    "Namangan", "Qashqadaryo", "Surxondaryo", "Xorazm", "Navoiy", "Jizzax",
    "Sirdaryo", "Qoraqalpoq", "Orol", "Zarafshon", "Vodiy", "Turon", "Osiyo",
    "Sharq", "G'arb", "Baraka", "Istiqlol", "Taraqqiyot", "Yangi Davr",
]

SECTOR_KEYWORDS = {
    "Issiqxona": ["Agro Greenhouse", "Limonariya Agro", "Tomat Export", "Yashil Voha", "Sabzavot Agro Invest"],
    "Paxta to'qimachilik klasteri": ["Cotton Cluster", "To'qimachilik Klasteri", "Paxta Sanoat", "Agro Textile", "Paxtakor Klaster"],
    "Qurilish materiallari": ["Sement Zavodi", "G'isht Invest", "Temir-Beton Konstruktsiyalari", "Qurilish Bazasi", "Stroy Material"],
    "Oziq-ovqat sanoati": ["Oziq-Ovqat Kombinati", "Sut Mahsulotlari", "Yog'-Moy", "Go'sht Sanoat", "Qandolat Fabrikasi"],
    "Oyna-shisha sanoati": ["Kvars Shisha", "Oyna Zavodi", "Glass Industries", "Shisha Tara", "Kvars Invest"],
    "Tekstil va tikuvchilik": ["Tekstil Fabrikasi", "Yigiruv-To'quv", "Tikuvchilik Majmuasi", "Apparel Group", "Moda Tekstil"],
    "Farmatsevtika": ["Dori-Darmon", "Farm Sanoat", "Bio Medikal", "Pharm Zavod", "Nika Farm Invest"],
    "Uzumchilik va agrosanoat": ["Uzum Agro", "Bog'dorchilik Klasteri", "Sharob Agrofirma", "Meva-Sabzavot", "Agro Eksport"],
    "Tibbiy va xizmat ko'rsatish": ["Klinika Tibbiyot", "Diagnostika Markazi", "Servis Logistika", "Tibbiy Xizmat", "Trans Servis"],
    "Avtomobil butlovchi qismlari": ["Avto Komponent", "Detal Motors", "Avto Sanoat Invest", "Mashinasozlik Zavodi", "Avto Plast"],
}

LEGAL_FORMS = ["MCHJ", "AJ", "XK", "Qo'shma Korxona (QK MCHJ)"]

OFFICER_NAMES = [
    "Akmal Rahimov", "Dilshod Karimov", "Bobur Alimov", "Sanjar Yusupov",
    "Jamshid Rustamov", "Otabek Saidov", "Shaxzod Tursunov", "Ulug'bek Nazarov",
    "Sherzod Qodirov", "Azizbek Mahmudov", "Farrux Ikromov", "Jahongir Vohidov",
    "Rustam Soliyev", "Nodirbek Hasanov", "Eldor Mirzayev", "Sardor Ahmedov",
    "Mansur Jo'rayev", "Bekzod Umarov",
]


@dataclass
class EnterpriseBorrower:
    """Enterprise borrower data representation."""
    inn: str
    company_name: str
    sector: str
    region: str
    branch_bxm: str
    officer_in_charge: str
    is_top20: bool
    loan_amount_uzs: float
    loan_balance_uzs: float
    interest_rate_pct: float
    collateral_value_uzs: float
    collateral_coverage_ratio: float
    revenue_current_uzs: float
    revenue_prev_month_uzs: float
    revenue_change_pct: float
    production_units_current: float
    production_units_prev: float
    production_change_pct: float
    electricity_kwh_current: float
    electricity_kwh_prev: float
    electricity_change_pct: float
    employee_count_current: int
    employee_count_prev: int
    ebitda_margin_pct: float
    net_profit_margin_pct: float
    current_ratio: float
    debt_to_equity: float
    dscr: float
    tax_arrears_uzs: float
    k2_card_index_amount_uzs: float
    overdue_days: int
    katm_credit_score: int
    credit_rating: str
    ground_truth_risk_tier: str
    monitoring_cycle: str
    last_monitoring_date: str


def generate_synthetic_portfolio(
    n_samples: int = PORTFOLIO_TOTAL_BORROWERS,
    top20_count: int = PORTFOLIO_TOP20_COUNT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic corporate credit monitoring dataset modeling Uzbek legal entities.

    Parameters:
        n_samples: Total number of borrowers (default: 603)
        top20_count: Number of TOP-20 large corporate borrowers (default: 20, capped at n_samples)
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame containing all enterprise profile, financial, and risk attributes.
    """
    rng = np.random.default_rng(seed)

    # Robust safety: ensure top20_count cannot exceed sample size
    effective_top20 = min(max(0, top20_count), n_samples)

    # Establish risk tier distribution: ~70% Green, ~20% Yellow, ~10% Red
    n_green = int(n_samples * 0.70)
    n_yellow = int(n_samples * 0.20)
    n_red = n_samples - n_green - n_yellow

    risk_tiers = (
        [RiskTier.GREEN.value] * n_green +
        [RiskTier.YELLOW.value] * n_yellow +
        [RiskTier.RED.value] * n_red
    )
    rng.shuffle(risk_tiers)

    # Top-20 indices selected from high-exposure borrowers
    if effective_top20 > 0:
        top20_indices = set(rng.choice(n_samples, size=effective_top20, replace=False))
    else:
        top20_indices = set()

    today = date(2026, 10, 1)
    records: List[dict] = []

    for i in range(n_samples):
        tier = risk_tiers[i]
        is_top20 = (i in top20_indices)

        # Unique 9-digit INN
        inn = str(300000000 + i + 1000)

        # Sector & Region
        sector = rng.choice(SECTORS)
        region = rng.choice(REGIONS)
        branch_id = rng.integers(1, 73)
        branch_bxm = f"{region.split()[0]} {branch_id:02d}-sonli BXM"
        officer = rng.choice(OFFICER_NAMES)

        # Company Name
        prefix = rng.choice(COMPANY_PREFIXES)
        keyword = rng.choice(SECTOR_KEYWORDS[sector])
        form = rng.choice(LEGAL_FORMS)
        company_name = f'"{prefix} {keyword}" {form}'

        # Loan sizing: TOP-20 vs mid-market corporate
        if is_top20:
            loan_amount_uzs = float(rng.uniform(60e9, 320e9))  # 60B - 320B UZS
        else:
            loan_amount_uzs = float(rng.uniform(4e9, 48e9))    # 4B - 48B UZS

        # Current balance is 50% - 95% of principal
        balance_ratio = rng.uniform(0.55, 0.95)
        loan_balance_uzs = float(loan_amount_uzs * balance_ratio)
        interest_rate_pct = float(rng.uniform(18.0, 24.5))

        # Macro shock + financial & operational metrics conditioned on ground truth tier
        macro_shock = float(rng.normal(0, 0.04))

        if tier == RiskTier.GREEN.value:
            # Healthy operational metrics with realistic variance
            revenue_change_pct = float(rng.normal(0.03, 0.11)) + macro_shock
            production_change_pct = float(revenue_change_pct * 0.70 + rng.normal(0, 0.08))
            electricity_change_pct = float(production_change_pct * 0.60 + rng.normal(0, 0.07))

            dscr = float(np.clip(rng.normal(1.58, 0.42), 0.75, 3.50))
            current_ratio = float(np.clip(rng.normal(1.48, 0.38), 0.75, 3.00))
            debt_to_equity = float(np.clip(rng.normal(1.75, 0.65), 0.50, 4.50))
            ebitda_margin_pct = float(np.clip(rng.normal(14.5, 5.5), -2.0, 32.0))
            net_profit_margin_pct = float(np.clip(rng.normal(7.8, 3.8), -3.0, 18.0))
            katm_score = int(np.clip(rng.normal(395, 42), 260, 495))
            overdue_days = int(rng.choice([0, 0, 0, rng.integers(1, 28)]))
            k2_amount = 0.0 if rng.random() > 0.08 else float(rng.uniform(1e6, 25e6))
            tax_arrears = 0.0 if rng.random() > 0.25 else float(rng.uniform(1e6, 30e6))
            collateral_coverage = float(np.clip(rng.normal(1.36, 0.26), 0.85, 2.50))
            emp_growth = float(rng.normal(0.02, 0.08))
            credit_rating = rng.choice(["AAA", "AA", "A", "BBB"])
            monitoring_cycle = "Har 6 oyda"

        elif tier == RiskTier.YELLOW.value:
            # Watch-list / emerging vulnerability
            revenue_change_pct = float(rng.normal(-0.07, 0.13)) + macro_shock
            production_change_pct = float(revenue_change_pct * 0.70 + rng.normal(0, 0.09))
            electricity_change_pct = float(production_change_pct * 0.60 + rng.normal(0, 0.08))

            dscr = float(np.clip(rng.normal(1.18, 0.34), 0.55, 2.20))
            current_ratio = float(np.clip(rng.normal(1.18, 0.30), 0.55, 2.20))
            debt_to_equity = float(np.clip(rng.normal(2.55, 0.85), 1.00, 5.50))
            ebitda_margin_pct = float(np.clip(rng.normal(8.5, 4.8), -5.0, 20.0))
            net_profit_margin_pct = float(rng.normal(3.5, 3.2))
            katm_score = int(np.clip(rng.normal(335, 48), 190, 440))
            overdue_days = int(rng.choice([
                rng.integers(8, 35),
                rng.integers(36, 75),
                rng.integers(20, 60),
            ]))
            k2_amount = 0.0 if rng.random() > 0.38 else float(rng.uniform(15e6, 450e6))
            tax_arrears = float(rng.uniform(10e6, 300e6))
            collateral_coverage = float(np.clip(rng.normal(1.15, 0.22), 0.70, 1.80))
            emp_growth = float(rng.normal(-0.04, 0.09))
            credit_rating = rng.choice(["BBB", "BB", "B"])
            monitoring_cycle = "Favqulodda (har oy)" if overdue_days > 45 else "Har chorakda"

        else:  # RED
            # Distressed / NPL with insolvency signs
            revenue_change_pct = float(rng.normal(-0.21, 0.17)) + macro_shock
            production_change_pct = float(revenue_change_pct * 0.70 + rng.normal(0, 0.11))
            electricity_change_pct = float(production_change_pct * 0.60 + rng.normal(0, 0.09))

            dscr = float(np.clip(rng.normal(0.85, 0.36), 0.20, 1.70))
            current_ratio = float(np.clip(rng.normal(0.90, 0.32), 0.20, 1.70))
            debt_to_equity = float(np.clip(rng.normal(3.85, 1.25), 1.50, 8.00))
            ebitda_margin_pct = float(np.clip(rng.normal(2.5, 6.0), -18.0, 16.0))
            net_profit_margin_pct = float(np.clip(rng.normal(-1.5, 4.8), -18.0, 9.0))
            katm_score = int(np.clip(rng.normal(255, 58), 100, 390))
            overdue_days = int(rng.choice([
                rng.integers(40, 85),
                rng.integers(86, 170),
                rng.integers(60, 130),
            ]))
            k2_amount = float(rng.uniform(100e6, 2.5e9)) if rng.random() > 0.22 else 0.0
            tax_arrears = float(rng.uniform(50e6, 1.2e9))
            collateral_coverage = float(np.clip(rng.normal(0.92, 0.24), 0.40, 1.60))
            emp_growth = float(rng.normal(-0.12, 0.11))
            credit_rating = rng.choice(["CCC", "CC", "D"])
            monitoring_cycle = "Favqulodda (har oy) / Sud jarayoni"

        # Realistic regulatory discretion divergence on ~7% borderline accounts
        assigned_tier = tier
        if rng.random() < 0.07:
            if tier == RiskTier.YELLOW.value:
                assigned_tier = rng.choice([RiskTier.GREEN.value, RiskTier.RED.value])
            elif tier == RiskTier.GREEN.value and (dscr < 1.15 or overdue_days > 15):
                assigned_tier = RiskTier.YELLOW.value
            elif tier == RiskTier.RED.value and (dscr > 1.10 and overdue_days < 70):
                assigned_tier = RiskTier.YELLOW.value

        tier = assigned_tier

        # Base operational figures
        collateral_value_uzs = float(loan_balance_uzs * collateral_coverage)
        prev_revenue_uzs = float(loan_balance_uzs * rng.uniform(0.20, 0.50))
        current_revenue_uzs = float(max(prev_revenue_uzs * (1.0 + revenue_change_pct), 1e7))

        base_units = float(rng.uniform(5000, 150000))
        prev_production_units = base_units
        current_production_units = float(max(base_units * (1.0 + production_change_pct), 10.0))

        base_kwh = float(rng.uniform(25000, 500000))
        prev_electricity_kwh = base_kwh
        current_electricity_kwh = float(max(base_kwh * (1.0 + electricity_change_pct), 500.0))

        prev_employees = int(rng.integers(25, 450))
        curr_employees = max(int(prev_employees * (1.0 + emp_growth)), 5)

        # Recent monitoring date (within last 180 days)
        days_ago = int(rng.integers(5, 175))
        last_monitoring_date = (today - timedelta(days=days_ago)).isoformat()

        borrower = EnterpriseBorrower(
            inn=inn,
            company_name=company_name,
            sector=sector,
            region=region,
            branch_bxm=branch_bxm,
            officer_in_charge=officer,
            is_top20=is_top20,
            loan_amount_uzs=round(loan_amount_uzs, 2),
            loan_balance_uzs=round(loan_balance_uzs, 2),
            interest_rate_pct=round(interest_rate_pct, 2),
            collateral_value_uzs=round(collateral_value_uzs, 2),
            collateral_coverage_ratio=round(collateral_coverage, 3),
            revenue_current_uzs=round(current_revenue_uzs, 2),
            revenue_prev_month_uzs=round(prev_revenue_uzs, 2),
            revenue_change_pct=round(revenue_change_pct, 4),
            production_units_current=round(current_production_units, 1),
            production_units_prev=round(prev_production_units, 1),
            production_change_pct=round(production_change_pct, 4),
            electricity_kwh_current=round(current_electricity_kwh, 1),
            electricity_kwh_prev=round(prev_electricity_kwh, 1),
            electricity_change_pct=round(electricity_change_pct, 4),
            employee_count_current=curr_employees,
            employee_count_prev=prev_employees,
            ebitda_margin_pct=round(ebitda_margin_pct, 2),
            net_profit_margin_pct=round(net_profit_margin_pct, 2),
            current_ratio=round(current_ratio, 3),
            debt_to_equity=round(debt_to_equity, 3),
            dscr=round(dscr, 3),
            tax_arrears_uzs=round(tax_arrears, 2),
            k2_card_index_amount_uzs=round(k2_amount, 2),
            overdue_days=overdue_days,
            katm_credit_score=katm_score,
            credit_rating=credit_rating,
            ground_truth_risk_tier=tier,
            monitoring_cycle=monitoring_cycle,
            last_monitoring_date=last_monitoring_date,
        )
        records.append(asdict(borrower))

    df = pd.DataFrame(records)
    return df
