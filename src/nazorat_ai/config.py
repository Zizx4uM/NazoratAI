"""
Configuration, domain constants, and thresholds for NazoratAI.
Implements Uzbek banking regulatory standards for corporate credit monitoring.
"""

from enum import Enum
from pathlib import Path
from typing import List, Dict, Any


class RiskTier(str, Enum):
    """Credit risk classification tiers (Central Bank of Uzbekistan classification)."""
    GREEN = "GREEN"    # Yashil / Standart (healthy, regular 6-month monitoring)
    YELLOW = "YELLOW"  # Sariq / Substandart (watch-list, overdue >45d, quarterly/emergency monitoring)
    RED = "RED"        # Qizil / Shubxali yoki Umidsiz (distressed / NPL >90d, legal action)


class EWSSeverity(str, Enum):
    """Early Warning System trigger severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Regulatory and operational thresholds
OVERDUE_DAYS_YELLOW_THRESHOLD: int = 45   # >45 days overdue -> Yellow trigger & emergency monitoring
OVERDUE_DAYS_RED_THRESHOLD: int = 90      # >90 days overdue -> Red / NPL trigger
REVENUE_DROP_MOM_THRESHOLD: float = -0.15 # >= 15% revenue drop month-over-month
PRODUCTION_DROP_MOM_THRESHOLD: float = -0.20 # >= 20% production output drop MoM
ELECTRICITY_DROP_MOM_THRESHOLD: float = -0.25 # >= 25% electricity consumption drop MoM
COLLATERAL_COVERAGE_THRESHOLD: float = 1.00 # Collateral value / Loan balance < 100%
DSCR_THRESHOLD: float = 1.10             # Debt Service Coverage Ratio < 1.10
K2_CARD_INDEX_THRESHOLD_UZS: float = 0.0  # Any positive 2-kartoteka arrears

# Target Portfolio Scope
PORTFOLIO_TOTAL_BORROWERS: int = 603
PORTFOLIO_TOP20_COUNT: int = 20

# 10 Key Industry Sectors in Uzbekistan Corporate Portfolio
SECTORS: List[str] = [
    "Issiqxona",
    "Paxta to'qimachilik klasteri",
    "Qurilish materiallari",
    "Oziq-ovqat sanoati",
    "Oyna-shisha sanoati",
    "Tekstil va tikuvchilik",
    "Farmatsevtika",
    "Uzumchilik va agrosanoat",
    "Tibbiy va xizmat ko'rsatish",
    "Avtomobil butlovchi qismlari",
]

# 14 Administrative Regions of Uzbekistan
REGIONS: List[str] = [
    "Toshkent shahri",
    "Toshkent viloyati",
    "Samarqand viloyati",
    "Farg'ona viloyati",
    "Andijon viloyati",
    "Namangan viloyati",
    "Buxoro viloyati",
    "Qashqadaryo viloyati",
    "Surxondaryo viloyati",
    "Xorazm viloyati",
    "Navoiy viloyati",
    "Jizzax viloyati",
    "Sirdaryo viloyati",
    "Qoraqalpog'iston Respublikasi",
]

# Monitoring Staff Allocation
HEAD_OFFICE_OFFICERS_COUNT: int = 18
BRANCH_BXM_COUNT: int = 72
TOTAL_MONITORING_STAFF: int = 90

# Default filesystem paths
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_MODEL_DIR = DEFAULT_BASE_DIR / "models"
DEFAULT_REPORT_DIR = DEFAULT_BASE_DIR / "reports"
