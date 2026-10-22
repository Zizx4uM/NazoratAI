"""
Feature engineering pipeline for corporate credit risk scoring.
Extracts financial ratios, operational proxies, and risk flags.
Fully robust against missing columns, nulls, and partial dictionaries.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from nazorat_ai.config import SECTORS


FEATURE_NAMES: List[str] = [
    "loan_balance_ratio",
    "collateral_coverage_ratio",
    "overdue_days",
    "has_k2_arrears",
    "k2_to_balance_ratio",
    "tax_arrears_to_balance_ratio",
    "revenue_change_pct",
    "production_change_pct",
    "electricity_change_pct",
    "employee_growth_pct",
    "dscr",
    "current_ratio",
    "debt_to_equity",
    "ebitda_margin_pct",
    "net_profit_margin_pct",
    "katm_credit_score",
    "interest_rate_pct",
    "is_top20",
] + [f"sector_{s}" for s in SECTORS]


def _get_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Safely extract a numeric pandas Series from DataFrame with reliable scalar fallback."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


class FeaturePipeline:
    """Transforms raw borrower records into model-ready numeric feature vectors."""

    def __init__(self):
        self.feature_names = list(FEATURE_NAMES)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract structured features from a raw portfolio DataFrame.
        Guaranteed to return all numeric columns without NaN and without crashing on missing columns.
        """
        features = pd.DataFrame(index=df.index)

        # 1. Balance to principal ratio
        loan_amt = _get_series(df, "loan_amount_uzs", 1.0).replace(0, 1.0)
        loan_bal = _get_series(df, "loan_balance_uzs", 0.0)
        features["loan_balance_ratio"] = (loan_bal / loan_amt).clip(0.0, 2.0)

        # 2. Collateral coverage ratio
        if "collateral_coverage_ratio" in df.columns:
            features["collateral_coverage_ratio"] = pd.to_numeric(
                df["collateral_coverage_ratio"], errors="coerce"
            ).fillna(1.0).clip(0.0, 5.0)
        else:
            collat_val = _get_series(df, "collateral_value_uzs", 0.0)
            safe_bal = loan_bal.replace(0, 1.0)
            features["collateral_coverage_ratio"] = (collat_val / safe_bal).clip(0.0, 5.0)

        # 3. Delinquency
        features["overdue_days"] = _get_series(df, "overdue_days", 0.0).clip(0, 720)

        # 4. K-2 card index arrears
        k2 = _get_series(df, "k2_card_index_amount_uzs", 0.0)
        features["has_k2_arrears"] = (k2 > 0).astype(float)
        safe_bal = loan_bal.replace(0, 1.0)
        features["k2_to_balance_ratio"] = (k2 / safe_bal).clip(0.0, 1.0)

        # 5. Tax arrears ratio
        tax = _get_series(df, "tax_arrears_uzs", 0.0)
        features["tax_arrears_to_balance_ratio"] = (tax / safe_bal).clip(0.0, 1.0)

        # 6. Trends (revenue, production, electricity)
        features["revenue_change_pct"] = _get_series(df, "revenue_change_pct", 0.0).clip(-1.0, 1.0)
        features["production_change_pct"] = _get_series(df, "production_change_pct", 0.0).clip(-1.0, 1.0)
        features["electricity_change_pct"] = _get_series(df, "electricity_change_pct", 0.0).clip(-1.0, 1.0)

        # 7. Employment growth
        curr_emp = _get_series(df, "employee_count_current", 50.0)
        prev_emp = _get_series(df, "employee_count_prev", 50.0).replace(0, 1.0)
        features["employee_growth_pct"] = ((curr_emp - prev_emp) / prev_emp).clip(-1.0, 2.0)

        # 8. Financial ratios
        features["dscr"] = _get_series(df, "dscr", 1.5).clip(0.0, 5.0)
        features["current_ratio"] = _get_series(df, "current_ratio", 1.5).clip(0.0, 10.0)
        features["debt_to_equity"] = _get_series(df, "debt_to_equity", 1.5).clip(0.0, 20.0)
        features["ebitda_margin_pct"] = _get_series(df, "ebitda_margin_pct", 15.0).clip(-50.0, 100.0)
        features["net_profit_margin_pct"] = _get_series(df, "net_profit_margin_pct", 10.0).clip(-50.0, 100.0)

        # 9. Bureau score and terms
        features["katm_credit_score"] = _get_series(df, "katm_credit_score", 400.0).clip(100, 500)
        features["interest_rate_pct"] = _get_series(df, "interest_rate_pct", 21.0).clip(5.0, 40.0)

        # is_top20 flag
        if "is_top20" in df.columns:
            features["is_top20"] = df["is_top20"].fillna(False).astype(float)
        else:
            features["is_top20"] = 0.0

        # 10. Sector one-hot encoding
        current_sector = df["sector"] if "sector" in df.columns else pd.Series("", index=df.index)
        for s in SECTORS:
            features[f"sector_{s}"] = (current_sector == s).astype(float)

        # Fill any lingering NaNs with 0.0
        features = features.fillna(0.0)

        # Ensure all columns exist in designated order
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0.0

        return features[self.feature_names]

    def extract_single_record(self, record: Dict) -> pd.DataFrame:
        """Helper to extract features from a single dictionary."""
        df_single = pd.DataFrame([record])
        return self.extract_features(df_single)
