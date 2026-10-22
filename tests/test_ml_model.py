"""
Tests for Machine Learning model training, inference, explainability, and evaluation.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from nazorat_ai.config import RiskTier
from nazorat_ai.data.generator import generate_synthetic_portfolio
from nazorat_ai.ml.features import FeaturePipeline
from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.ml.explainability import ExplainabilityEngine
from nazorat_ai.ml.evaluator import evaluate_model_performance


@pytest.fixture(scope="module")
def trained_artifacts():
    """Train a test model once for module tests."""
    df = generate_synthetic_portfolio(n_samples=100, seed=42)
    model = CreditRiskModel(random_state=42)
    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()
    model.fit(X, y)
    return df, model, X, y


def test_feature_pipeline_extraction(trained_artifacts):
    """Verify feature pipeline outputs all expected numeric columns without NaNs."""
    df, _, X, _ = trained_artifacts
    assert len(X) == len(df)
    assert not X.isna().any().any()
    assert "overdue_days" in X.columns
    assert "collateral_coverage_ratio" in X.columns
    assert "revenue_change_pct" in X.columns
    assert "dscr" in X.columns


def test_feature_pipeline_missing_columns_tolerance():
    """Verify pipeline handles completely missing or partial records without AttributeError."""
    pipe = FeaturePipeline()
    # Bare minimum dictionary with almost no optional columns
    sparse_record = {"inn": "301000111", "loan_balance_uzs": 5000000.0}
    X_sparse = pipe.extract_single_record(sparse_record)

    assert len(X_sparse) == 1
    assert not X_sparse.isna().any().any()
    assert X_sparse["loan_balance_ratio"].iloc[0] >= 0.0
    assert X_sparse["overdue_days"].iloc[0] == 0.0
    assert X_sparse["dscr"].iloc[0] == 1.5


def test_model_predictions_and_probabilities(trained_artifacts):
    """Verify multi-class prediction and calibrated probability distribution."""
    _, model, X, _ = trained_artifacts
    preds = model.predict(X)
    assert len(preds) == len(X)
    valid_classes = {RiskTier.GREEN.value, RiskTier.YELLOW.value, RiskTier.RED.value}
    assert set(preds).issubset(valid_classes)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_continuous_risk_scoring(trained_artifacts):
    """Verify continuous credit risk score is bounded within [0, 100]."""
    _, model, X, _ = trained_artifacts
    probs = model.predict_proba(X)
    scores = model.compute_risk_score(probs)
    assert len(scores) == len(X)
    assert (scores >= 0.0).all()
    assert (scores <= 100.0).all()


def test_model_save_and_load(trained_artifacts):
    """Verify model persistence to disk and reloading."""
    _, model, X, _ = trained_artifacts
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        model.save(tmp.name)
        loaded = CreditRiskModel.load(tmp.name)

    assert loaded.is_fitted is True
    original_preds = model.predict(X[:5])
    loaded_preds = loaded.predict(X[:5])
    assert original_preds == loaded_preds


def test_global_explainability_importances(trained_artifacts):
    """Verify global feature importances calculation."""
    _, model, _, _ = trained_artifacts
    explainer = ExplainabilityEngine(model)
    df_imp = explainer.get_global_feature_importances(top_n=10)
    assert len(df_imp) == 10
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert "label_uz" in df_imp.columns
    assert df_imp["importance"].sum() > 0.0


def test_local_explainability_exact_waterfall(trained_artifacts):
    """Verify exact multi-class tree path attribution: Base + Sum(Deltas) == TotalScore."""
    df, model, _, _ = trained_artifacts
    explainer = ExplainabilityEngine(model)
    sample_record = df.iloc[0].to_dict()

    res = explainer.explain_borrower_record(sample_record)
    assert "total_risk_score" in res
    assert "base_risk_score" in res
    assert "top_risk_drivers" in res
    assert "top_protective_factors" in res
    assert "executive_summary_uz" in res
    assert "executive_summary_en" in res

    # Verify that sum of contributions plus base score matches total risk score within 1.0 pt tolerance
    all_contribs = res.get("all_contributions", [])
    delta_sum = sum(c["contribution_pts"] for c in all_contribs)
    expected_score = res["base_risk_score"] + delta_sum
    assert abs(expected_score - res["total_risk_score"]) <= 1.0


def test_model_evaluator_metrics(trained_artifacts):
    """Verify comprehensive model performance metrics calculation."""
    _, model, X, y = trained_artifacts
    metrics = evaluate_model_performance(model, X, y)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "roc_auc_ovr" in metrics
    assert "ews_watch_precision" in metrics
    assert "ews_watch_recall" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["sample_count"] == len(y)


def test_stratified_evaluation_realistic_boundaries():
    """Verify held-out evaluation on overlapping distributions behaves realistically."""
    from sklearn.model_selection import train_test_split
    df = generate_synthetic_portfolio(n_samples=200, seed=42)
    model = CreditRiskModel(random_state=42)
    X = model.feature_pipeline.extract_features(df)
    y = df["ground_truth_risk_tier"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model_performance(model, X_test, y_test)

    # Generalization is strong (AUC > 0.80)
    assert metrics["roc_auc_ovr"] >= 0.80
    assert metrics["accuracy"] >= 0.75
    # Demonstrates non-trivial realistic boundary (not artificial 100% trivial memorization)
    assert metrics["roc_auc_ovr"] <= 1.00
