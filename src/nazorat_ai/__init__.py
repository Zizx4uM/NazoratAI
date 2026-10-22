"""
NazoratAI: Banking Corporate Credit Monitoring and Early Warning System (EWS).
Designed for commercial banks in Uzbekistan.
"""

__version__ = "1.0.0"
__author__ = "NazoratAI Engineering Team"

from nazorat_ai.config import RiskTier, EWSSeverity
from nazorat_ai.ews.rules import EWSEngine
from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.ml.explainability import ExplainabilityEngine
from nazorat_ai.document_review.analyzer import HybridDocumentReviewer
from nazorat_ai.reporting.generator import ReportGenerator

__all__ = [
    "RiskTier",
    "EWSSeverity",
    "EWSEngine",
    "CreditRiskModel",
    "ExplainabilityEngine",
    "HybridDocumentReviewer",
    "ReportGenerator",
]
