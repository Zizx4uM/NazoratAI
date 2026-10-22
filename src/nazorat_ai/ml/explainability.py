"""
Explainability Engine for NazoratAI Credit Risk Scoring.
Provides:
1. Global feature importances (Mean Decrease in Impurity / Permutation).
2. Local instance-level feature attributions (Tree Path Decomposition / Saabas method & exact additive score waterfall).
3. Human-readable bilingual (Uzbek & English) risk narrative for credit inspectors and board reports.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from nazorat_ai.ml.model import CreditRiskModel
from nazorat_ai.config import SECTORS


# Friendly bilingual translations and interpretations for features
FEATURE_DICTIONARY: Dict[str, Dict[str, str]] = {
    "overdue_days": {
        "uz": "Kredit to'lovi muddati kechikishi (kunlar soni)",
        "en": "Loan Delinquency / Overdue Days",
    },
    "revenue_change_pct": {
        "uz": "Oylik tushum o'zgarishi (MoM %)",
        "en": "Revenue Growth / Slump (MoM %)",
    },
    "production_change_pct": {
        "uz": "Ishlab chiqarish hajmi o'zgarishi (%)",
        "en": "Production Output Change (%)",
    },
    "collateral_coverage_ratio": {
        "uz": "Garov ta'minoti qoplami koeffitsienti",
        "en": "Collateral Coverage Ratio",
    },
    "k2_to_balance_ratio": {
        "uz": "2-kartoteka qarzdorligi ulushi",
        "en": "Card Index 2 Arrears Ratio",
    },
    "has_k2_arrears": {
        "uz": "2-kartoteka mavjudligi",
        "en": "Presence of K-2 Card Index Arrears",
    },
    "dscr": {
        "uz": "Qarzga xizmat ko'rsatish koeffitsienti (DSCR)",
        "en": "Debt Service Coverage Ratio (DSCR)",
    },
    "electricity_change_pct": {
        "uz": "Elektr energiyasi iste'moli dinamikasi (%)",
        "en": "Electricity Consumption Change (%)",
    },
    "current_ratio": {
        "uz": "Joriy likvidlik koeffitsienti",
        "en": "Current Ratio (Liquidity)",
    },
    "debt_to_equity": {
        "uz": "Qarz / Xususiy kapital (Leverage)",
        "en": "Debt-to-Equity Ratio",
    },
    "katm_credit_score": {
        "uz": "KATM kredit reyting skori",
        "en": "Credit Bureau (KATM) Score",
    },
    "tax_arrears_to_balance_ratio": {
        "uz": "Soliq qarzdorligi ulushi",
        "en": "Tax Arrears to Loan Ratio",
    },
    "loan_balance_ratio": {
        "uz": "Kredit qoldig'ining asosiy qarzga nisbati",
        "en": "Loan Balance to Principal Ratio",
    },
    "ebitda_margin_pct": {
        "uz": "EBITDA rentabelligi (%)",
        "en": "EBITDA Margin (%)",
    },
    "net_profit_margin_pct": {
        "uz": "Sof foyda rentabelligi (%)",
        "en": "Net Profit Margin (%)",
    },
    "employee_growth_pct": {
        "uz": "Xodimlar soni o'zgarishi (%)",
        "en": "Employee Headcount Growth (%)",
    },
    "is_top20": {
        "uz": "Yirik qarz oluvchi (TOP-20)",
        "en": "TOP-20 Corporate Borrower Exposure",
    },
    "interest_rate_pct": {
        "uz": "Kredit foiz stavkasi (%)",
        "en": "Interest Rate (%)",
    },
}

# Populate sector translations dynamically
for s in SECTORS:
    FEATURE_DICTIONARY[f"sector_{s}"] = {
        "uz": f"Soha: {s}",
        "en": f"Sector: {s}",
    }


@dataclass
class FeatureContribution:
    """Represents a single feature's contribution to risk score."""
    feature_name: str
    feature_label_uz: str
    feature_label_en: str
    feature_value: float
    contribution_pts: float  # Points added (+) or subtracted (-) from base risk score
    impact_direction: str    # "RISK_INCREASING" or "PROTECTIVE"
    explanation_uz: str
    explanation_en: str


class ExplainabilityEngine:
    """
    Computes global feature importances and local instance attributions.
    """

    def __init__(self, model: CreditRiskModel):
        self.model = model
        if not model.is_fitted:
            raise RuntimeError("Model must be fitted before initializing ExplainabilityEngine.")
        self.feature_names = model.feature_pipeline.feature_names

    def get_global_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Return top N features by Mean Decrease in Impurity (Random Forest feature importance).
        """
        rf = self.model.classifier
        importances = rf.feature_importances_
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        df_imp["label_uz"] = df_imp["feature"].apply(
            lambda f: FEATURE_DICTIONARY.get(f, {}).get("uz", f)
        )
        df_imp["label_en"] = df_imp["feature"].apply(
            lambda f: FEATURE_DICTIONARY.get(f, {}).get("en", f)
        )
        return df_imp.head(top_n)

    def explain_borrower_record(
        self,
        record: Dict,
        top_k: int = 6,
    ) -> Dict:
        """
        Decomposes the model prediction for a single borrower using Tree Path Attribution.
        Computes exact additive multi-class score decomposition:
            RiskScore = BaseScore + sum_j(contribution_j)
        where RiskScore = 10 * P(Green) + 50 * P(Yellow) + 95 * P(Red).

        Parameters:
            record: Dict of borrower features
            top_k: Number of key positive and negative drivers to highlight

        Returns:
            Structured explanation dictionary with waterfall data and bilingual summary.
        """
        X = self.model.feature_pipeline.extract_single_record(record)
        rf = self.model.classifier
        X_val = X.values[0]

        n_features = len(self.feature_names)
        contributions = np.zeros(n_features)
        base_value = 0.0

        # Weights corresponding to [Green (0), Yellow (1), Red (2)] in continuous score
        score_weights = np.array([10.0, 50.0, 95.0])

        # Iterate over individual trees in the forest (Tree Path Decomposition)
        for tree in rf.estimators_:
            tree_model = tree.tree_
            # Leaf values are class counts -> normalize to probability
            node_counts = tree_model.value[:, 0, :]
            # Shape: (n_nodes, n_classes)
            node_probs = node_counts / np.sum(node_counts, axis=1, keepdims=True)

            # Node expected risk score: dot product with score weights
            node_scores = np.dot(node_probs, score_weights)

            # Root expected value
            base_value += node_scores[0]

            # Trace path for sample
            node_id = 0
            while tree_model.children_left[node_id] != tree_model.children_right[node_id]:
                feature_idx = tree_model.feature[node_id]
                threshold = tree_model.threshold[node_id]

                if X_val[feature_idx] <= threshold:
                    next_node = tree_model.children_left[node_id]
                else:
                    next_node = tree_model.children_right[node_id]

                # Exact score delta attributed to split feature
                delta = node_scores[next_node] - node_scores[node_id]
                contributions[feature_idx] += delta
                node_id = next_node

        n_trees = len(rf.estimators_)
        base_value /= n_trees
        contributions /= n_trees

        feature_items: List[FeatureContribution] = []
        for i, feat_name in enumerate(self.feature_names):
            pts = contributions[i]
            if abs(pts) < 0.05:
                continue

            val = float(X_val[i])
            label_uz = FEATURE_DICTIONARY.get(feat_name, {}).get("uz", feat_name)
            label_en = FEATURE_DICTIONARY.get(feat_name, {}).get("en", feat_name)

            if pts > 0:
                direction = "RISK_INCREASING"
                expl_uz = f"{label_uz} ({val:,.2f}) risk darajasini +{pts:.1f} ballga oshirmoqda"
                expl_en = f"{label_en} ({val:,.2f}) increases risk score by +{pts:.1f} pts"
            else:
                direction = "PROTECTIVE"
                expl_uz = f"{label_uz} ({val:,.2f}) xatarni kamaytirib, -{abs(pts):.1f} ball pasaytirmoqda"
                expl_en = f"{label_en} ({val:,.2f}) protects portfolio, reducing risk by -{abs(pts):.1f} pts"

            feature_items.append(FeatureContribution(
                feature_name=feat_name,
                feature_label_uz=label_uz,
                feature_label_en=label_en,
                feature_value=val,
                contribution_pts=round(float(pts), 2),
                impact_direction=direction,
                explanation_uz=expl_uz,
                explanation_en=expl_en,
            ))

        # Sort by absolute impact
        feature_items.sort(key=lambda x: abs(x.contribution_pts), reverse=True)

        top_risk_drivers = [f for f in feature_items if f.impact_direction == "RISK_INCREASING"][:top_k]
        top_protective = [f for f in feature_items if f.impact_direction == "PROTECTIVE"][:top_k]

        pred_tier = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        total_risk_score = float(self.model.compute_risk_score(probs.reshape(1, -1))[0])

        # Bilingual executive summary synthesis
        if top_risk_drivers:
            lead_factor = top_risk_drivers[0]
            summary_uz = (
                f"Mijoz bo'yicha eng katta xatar omili: {lead_factor.feature_label_uz} "
                f"(ko'rsatkich: {lead_factor.feature_value:,.2f}, qo'shilgan risk: +{lead_factor.contribution_pts:.1f} ball). "
                f"Umumiy ML risk skori: {total_risk_score:.1f} / 100 ({pred_tier})."
            )
            summary_en = (
                f"Primary credit risk driver: {lead_factor.feature_label_en} "
                f"(value: {lead_factor.feature_value:,.2f}, impact: +{lead_factor.contribution_pts:.1f} pts). "
                f"Overall ML risk score: {total_risk_score:.1f} / 100 ({pred_tier})."
            )
        else:
            summary_uz = f"Xavf omillari aniqlanmadi. Mijoz barqaror moliyaviy holatda ({pred_tier})."
            summary_en = f"No significant risk drivers detected. Borrower maintains stable credit profile ({pred_tier})."

        return {
            "predicted_risk_tier": pred_tier,
            "total_risk_score": round(total_risk_score, 1),
            "base_risk_score": round(base_value, 1),
            "top_risk_drivers": [asdict_contrib(c) for c in top_risk_drivers],
            "top_protective_factors": [asdict_contrib(c) for c in top_protective],
            "all_contributions": [asdict_contrib(c) for c in feature_items],
            "executive_summary_uz": summary_uz,
            "executive_summary_en": summary_en,
        }


def asdict_contrib(c: FeatureContribution) -> Dict:
    return {
        "feature_name": c.feature_name,
        "feature_label_uz": c.feature_label_uz,
        "feature_label_en": c.feature_label_en,
        "feature_value": c.feature_value,
        "contribution_pts": c.contribution_pts,
        "impact_direction": c.impact_direction,
        "explanation_uz": c.explanation_uz,
        "explanation_en": c.explanation_en,
    }
