"""
Model evaluation and validation metrics for NazoratAI.
Calculates multi-class classification metrics, confusion matrix, ROC-AUC,
and regulatory EWS precision/recall against target benchmarks.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from nazorat_ai.config import RiskTier
from nazorat_ai.ml.model import CreditRiskModel


def evaluate_model_performance(
    model: CreditRiskModel,
    X_test: pd.DataFrame,
    y_test: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive credit risk evaluation metrics.
    """
    y_true = list(y_test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    classes = [RiskTier.GREEN.value, RiskTier.YELLOW.value, RiskTier.RED.value]

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class breakdown
    per_class = {}
    for cls in classes:
        cls_prec = float(precision_score(y_true, y_pred, labels=[cls], average="micro", zero_division=0))
        cls_rec = float(recall_score(y_true, y_pred, labels=[cls], average="micro", zero_division=0))
        cls_f1 = float(f1_score(y_true, y_pred, labels=[cls], average="micro", zero_division=0))
        per_class[cls] = {
            "precision": round(cls_prec, 3),
            "recall": round(cls_rec, 3),
            "f1_score": round(cls_f1, 3),
            "support": sum(1 for y in y_true if y == cls),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_dict = {
        "labels": classes,
        "matrix": cm.tolist(),
    }

    # Multi-class ROC-AUC (OVR)
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == y_proba.shape[1]:
            roc_auc = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
        else:
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0

    # Specific EWS KPIs required by bank mandate:
    # Yellow/Red precision (Watchlist + Distressed detection rate)
    binary_true = [1 if y in (RiskTier.YELLOW.value, RiskTier.RED.value) else 0 for y in y_true]
    binary_pred = [1 if y in (RiskTier.YELLOW.value, RiskTier.RED.value) else 0 for y in y_pred]

    ews_prec = float(precision_score(binary_true, binary_pred, zero_division=0))
    ews_rec = float(recall_score(binary_true, binary_pred, zero_division=0))

    # Benchmark compliance checks (Doc TZ requires Precision >= 70%, Recall >= 65%)
    meets_precision_target = bool(ews_prec >= 0.70)
    meets_recall_target = bool(ews_rec >= 0.65)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_ovr": round(roc_auc, 4),
        "ews_watch_precision": round(ews_prec, 4),
        "ews_watch_recall": round(ews_rec, 4),
        "kpi_precision_met": meets_precision_target,
        "kpi_recall_met": meets_recall_target,
        "per_class_metrics": per_class,
        "confusion_matrix": cm_dict,
        "sample_count": len(y_true),
    }
