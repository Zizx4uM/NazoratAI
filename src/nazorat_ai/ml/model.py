"""
Machine Learning Credit Risk Classifier Model for NazoratAI.
Predicts risk tier (GREEN, YELLOW, RED), calibrated class probabilities,
and a 0-100 continuous credit risk score.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from nazorat_ai.config import RiskTier
from nazorat_ai.ml.features import FeaturePipeline


LABEL_TO_INT = {
    RiskTier.GREEN.value: 0,
    RiskTier.YELLOW.value: 1,
    RiskTier.RED.value: 2,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


class CreditRiskModel:
    """
    Supervised classifier for multi-class corporate borrower risk tier prediction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_pipeline = FeaturePipeline()
        self.classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        )
        self.is_fitted = False
        self.classes_ = [RiskTier.GREEN.value, RiskTier.YELLOW.value, RiskTier.RED.value]

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, List[str]]) -> "CreditRiskModel":
        """
        Train classifier on features X and risk tier labels y.
        """
        y_series = pd.Series(y)
        # Convert string labels to integers if needed
        if y_series.dtype == object or isinstance(y_series.iloc[0], str):
            y_encoded = y_series.map(LABEL_TO_INT).fillna(0).astype(int)
        else:
            y_encoded = y_series.astype(int)

        self.classifier.fit(X, y_encoded)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        Predict risk tier for input feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        y_pred = self.classifier.predict(X)
        return [INT_TO_LABEL.get(val, RiskTier.GREEN.value) for val in y_pred]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated probabilities for [GREEN, YELLOW, RED].
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        probs = self.classifier.predict_proba(X)
        # Handle cases where not all classes appeared in training batch
        if probs.shape[1] < 3:
            full_probs = np.zeros((len(X), 3))
            for i, cls_idx in enumerate(self.classifier.classes_):
                full_probs[:, cls_idx] = probs[:, i]
            return full_probs
        return probs

    def compute_risk_score(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute continuous credit risk score between 0 (safest) and 100 (highest risk).
        Score formula: 10 * P(Green) + 50 * P(Yellow) + 95 * P(Red)
        """
        # Weights for [Green, Yellow, Red]
        weights = np.array([10.0, 50.0, 95.0])
        scores = np.dot(probs, weights)
        return np.clip(scores, 0.0, 100.0)

    def evaluate_borrower_record(self, record: Dict) -> Dict:
        """
        End-to-end scoring of a single borrower record dictionary.
        """
        X = self.feature_pipeline.extract_single_record(record)
        pred_tier = self.predict(X)[0]
        probs = self.predict_proba(X)[0]
        risk_score = float(self.compute_risk_score(probs.reshape(1, -1))[0])

        return {
            "predicted_risk_tier": pred_tier,
            "risk_score_0_100": round(risk_score, 1),
            "prob_green": round(float(probs[0]), 3),
            "prob_yellow": round(float(probs[1]), 3),
            "prob_red": round(float(probs[2]), 3),
            "features_extracted": X.iloc[0].to_dict(),
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Save model artifact to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "classifier": self.classifier,
            "is_fitted": self.is_fitted,
            "random_state": self.random_state,
            "classes_": self.classes_,
        }, path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "CreditRiskModel":
        """Load model artifact from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        data = joblib.load(path)
        model = cls(random_state=data.get("random_state", 42))
        model.classifier = data["classifier"]
        model.is_fitted = data["is_fitted"]
        model.classes_ = data.get("classes_", [RiskTier.GREEN.value, RiskTier.YELLOW.value, RiskTier.RED.value])
        return model
