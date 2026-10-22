"""
Tests for portfolio dataset generation, schema validation, and loading.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from nazorat_ai.config import RiskTier, SECTORS, REGIONS
from nazorat_ai.data.generator import generate_synthetic_portfolio
from nazorat_ai.data.loader import load_portfolio_data, validate_portfolio_dataframe


def test_portfolio_generation_shape_and_counts():
    """Verify default generation outputs exactly 603 corporate borrowers and 20 TOP-20s."""
    df = generate_synthetic_portfolio(n_samples=603, top20_count=20, seed=42)
    assert len(df) == 603
    assert df["is_top20"].sum() == 20

    # Verify unique INNs
    assert df["inn"].nunique() == 603

    # Check risk tier distribution
    tiers = df["ground_truth_risk_tier"].value_counts()
    assert RiskTier.GREEN.value in tiers
    assert RiskTier.YELLOW.value in tiers
    assert RiskTier.RED.value in tiers
    # Green should be largest class (~70%)
    assert tiers[RiskTier.GREEN.value] > 380


def test_portfolio_sectors_and_regions():
    """Verify coverage of designated sectors and regions."""
    df = generate_synthetic_portfolio(n_samples=603, seed=42)
    unique_sectors = set(df["sector"].unique())
    assert unique_sectors.issubset(set(SECTORS))
    assert len(unique_sectors) >= 8

    unique_regions = set(df["region"].unique())
    assert unique_regions.issubset(set(REGIONS))
    assert len(unique_regions) >= 10


def test_portfolio_small_sample_bounds():
    """Verify generator does not crash when n_samples < top20_count."""
    df_small = generate_synthetic_portfolio(n_samples=5, top20_count=20, seed=123)
    assert len(df_small) == 5
    assert df_small["is_top20"].sum() == 5  # Capped at n_samples

    df_zero = generate_synthetic_portfolio(n_samples=0, top20_count=10, seed=1)
    assert len(df_zero) == 0


def test_schema_validation_success():
    """Verify that generated data passes all schema and business validation rules."""
    df = generate_synthetic_portfolio(n_samples=50, seed=123)
    is_valid, issues = validate_portfolio_dataframe(df)
    assert is_valid is True
    assert len(issues) == 0


def test_schema_validation_failure_on_missing_cols():
    """Verify schema validator detects missing columns."""
    df = pd.DataFrame({"company_name": ["Test MCHJ"]})
    is_valid, issues = validate_portfolio_dataframe(df)
    assert is_valid is False
    assert any("Missing required columns" in issue for issue in issues)


def test_schema_validation_empty_dataframe():
    """Verify schema validator detects empty DataFrames."""
    df_empty = pd.DataFrame()
    is_valid, issues = validate_portfolio_dataframe(df_empty)
    assert is_valid is False
    assert any("empty" in issue.lower() for issue in issues)


def test_data_loader_csv_roundtrip():
    """Verify CSV export and reload with proper typing."""
    df = generate_synthetic_portfolio(n_samples=25, seed=99)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        loaded_df = load_portfolio_data(tmp.name)

    assert len(loaded_df) == 25
    assert isinstance(loaded_df["inn"].iloc[0], str)
    assert loaded_df["loan_balance_uzs"].dtype in (float, "float64")
    assert loaded_df["is_top20"].dtype in (bool, "bool")
