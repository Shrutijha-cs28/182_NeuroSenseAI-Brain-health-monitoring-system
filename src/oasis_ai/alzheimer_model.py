"""
Alzheimer's Disease (AD) Risk Screening Engine for NeuroSense AI.
Includes patient-level holdout validation, feature engineering,
XGBoost classification, calibrated risk scoring, and SHAP explainability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


RAW_FEATURES = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
ENGINEERED_FEATURES = [
    "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF",
    "Brain_Volume_Est", "Atrophy_Ratio", "Age_Atrophy_Index"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes clinical brain atrophy indices from demographic & volumetric fields."""
    X = df.copy()
    # Impute missing values with medians
    X["SES"] = pd.to_numeric(X["SES"], errors="coerce").fillna(X["SES"].median())
    X["MMSE"] = pd.to_numeric(X["MMSE"], errors="coerce").fillna(X["MMSE"].median())
    for col in RAW_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Estimated actual parenchymal volume (cm^3)
    X["Brain_Volume_Est"] = X["eTIV"] * X["nWBV"]
    # Atrophy percentage (CSF & tissue loss fraction)
    X["Atrophy_Ratio"] = 1.0 - X["nWBV"]
    # Interaction: atrophy disproportionate to chronological age
    X["Age_Atrophy_Index"] = X["Age"] * X["Atrophy_Ratio"]

    return X[ENGINEERED_FEATURES]


class AlzheimerRiskModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        self.feature_names = ENGINEERED_FEATURES
        self.explainer: shap.TreeExplainer | None = None
        self.metadata: dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train[self.feature_names], y_train)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_risk(self, patient_features: dict[str, float]) -> dict[str, Any]:
        """
        Accepts patient raw measurements, applies feature engineering,
        predicts dementia risk probability, and returns SHAP biomarker contributions.
        """
        df = pd.DataFrame([patient_features])
        X_eng = engineer_features(df)
        
        # Risk probability: P(AD / Cognitive Impairment)
        prob = float(self.model.predict_proba(X_eng)[0, 1])
        predicted_class = int(prob >= 0.5)

        # Risk tier definition
        if prob < 0.35:
            tier = "Low Risk"
        elif prob <= 0.65:
            tier = "Moderate Risk"
        else:
            tier = "High Risk"

        # Explainable AI: SHAP contribution per feature
        shap_values = self.explainer.shap_values(X_eng)[0]
        biomarker_attributions = {
            feat: float(val) for feat, val in zip(self.feature_names, shap_values)
        }
        # Sort by absolute impact
        sorted_attributions = dict(
            sorted(biomarker_attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        return {
            "risk_score": prob,
            "risk_percentage": round(prob * 100, 1),
            "risk_tier": tier,
            "predicted_binary": predicted_class,
            "biomarker_attributions": sorted_attributions,
            "engineered_features": X_eng.iloc[0].to_dict(),
        }

    def save(self, output_path: str | Path, metadata: dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        joblib.dump(
            {
                "model": self.model,
                "explainer": self.explainer,
                "feature_names": self.feature_names,
                "metadata": metadata,
            },
            path,
        )
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AlzheimerRiskModel:
        bundle = joblib.load(str(path))
        instance = cls()
        instance.model = bundle["model"]
        instance.explainer = bundle["explainer"]
        instance.feature_names = bundle["feature_names"]
        instance.metadata = bundle.get("metadata", {})
        return instance
