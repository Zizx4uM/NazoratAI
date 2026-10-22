"""
Data loading and validation module for NazoratAI.
Supports CSV, Excel (.xlsx, .xls), and JSON ingestion with strict schema checks.
"""

from pathlib import Path
from typing import Union, List, Tuple
import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "inn",
    "company_name",
    "sector",
    "region",
    "loan_amount_uzs",
    "loan_balance_uzs",
    "overdue_days",
    "collateral_coverage_ratio",
    "revenue_change_pct",
    "production_change_pct",
    "dscr",
]


def load_portfolio_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load corporate credit monitoring portfolio data from CSV, Excel, or JSON.

    Parameters:
        file_path: Path to the data file.

    Returns:
        pd.DataFrame containing sanitized portfolio data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .xlsx, .xls, .json, .parquet")

    # Ensure INN is treated as a string
    if "inn" in df.columns:
        df["inn"] = df["inn"].astype(str)

    # Convert numeric fields
    numeric_cols = [
        "loan_amount_uzs", "loan_balance_uzs", "collateral_value_uzs",
        "collateral_coverage_ratio", "revenue_current_uzs", "revenue_prev_month_uzs",
        "revenue_change_pct", "production_units_current", "production_units_prev",
        "production_change_pct", "electricity_kwh_current", "electricity_kwh_prev",
        "electricity_change_pct", "ebitda_margin_pct", "net_profit_margin_pct",
        "current_ratio", "debt_to_equity", "dscr", "tax_arrears_uzs",
        "k2_card_index_amount_uzs", "overdue_days", "katm_credit_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Convert boolean flags
    if "is_top20" in df.columns:
        df["is_top20"] = df["is_top20"].astype(bool)

    return df


def validate_portfolio_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame conforms to the expected portfolio schema.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []

    if len(df) == 0:
        issues.append("Portfolio DataFrame is empty (0 rows)")
        return (False, issues)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")

    if "inn" in df.columns:
        duplicate_inns = df["inn"].duplicated().sum()
        if duplicate_inns > 0:
            issues.append(f"Found {duplicate_inns} duplicate INN values")

    if "loan_balance_uzs" in df.columns:
        negative_balances = (df["loan_balance_uzs"] < 0).sum()
        if negative_balances > 0:
            issues.append(f"Found {negative_balances} records with negative loan balance")

    if "overdue_days" in df.columns:
        invalid_overdue = (df["overdue_days"] < 0).sum()
        if invalid_overdue > 0:
            issues.append(f"Found {invalid_overdue} records with negative overdue days")

    return (len(issues) == 0, issues)
