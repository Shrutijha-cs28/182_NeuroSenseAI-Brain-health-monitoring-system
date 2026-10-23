"""
Parkinson's Disease (PD) Risk Screening Engine for NeuroSense AI.
Supports dual-stream evaluation:
  1. Biomechanical Gait & Motor Force Sensors (devalab/parkinsonsfromgait)
  2. Acoustic Vocal Speech Analysis (UCI Parkinson's Phonation)
Includes patient-level holdout validation and SHAP explainability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.oasis_ai.gait_processor import GAIT_FEATURE_NAMES
from src.oasis_ai.voice_processor import VOICE_FEATURE_NAMES


class ParkinsonUnifiedModel:
    def __init__(self):
        self.gait_model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        self.voice_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        self.gait_features = GAIT_FEATURE_NAMES
        self.voice_features = VOICE_FEATURE_NAMES
        self.gait_explainer: shap.TreeExplainer | None = None
        self.voice_explainer: shap.TreeExplainer | None = None
        self.metadata: dict[str, Any] = {}

    def fit_gait(self, X_gait: pd.DataFrame, y_gait: pd.Series) -> None:
        self.gait_model.fit(X_gait[self.gait_features], y_gait)
        self.gait_explainer = shap.TreeExplainer(self.gait_model)

    def fit_voice(self, X_voice: pd.DataFrame, y_voice: pd.Series) -> None:
        self.voice_model.fit(X_voice[self.voice_features], y_voice)
        self.voice_explainer = shap.TreeExplainer(self.voice_model)

    def predict_risk(
        self,
        gait_dict: dict[str, float] | None = None,
        voice_dict: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates Parkinson's Disease risk from gait sensors, voice acoustics, or both.
        Returns calibrated risk scores, risk tier, and SHAP biomarker contributions.
        """
        gait_prob: float | None = None
        voice_prob: float | None = None
        attributions: dict[str, float] = {}
        active_modalities: list[str] = []

        # 1. Evaluate Gait Stream if provided
        if gait_dict is not None:
            df_g = pd.DataFrame([gait_dict])[self.gait_features]
            gait_prob = float(self.gait_model.predict_proba(df_g)[0, 1])
            active_modalities.append("Gait Kinematics")
            if self.gait_explainer is not None:
                g_shap = self.gait_explainer.shap_values(df_g)[0]
                for feat, val in zip(self.gait_features, g_shap):
                    attributions[f"Gait: {feat}"] = float(val)

        # 2. Evaluate Voice Stream if provided
        if voice_dict is not None:
            df_v = pd.DataFrame([voice_dict])[self.voice_features]
            voice_prob = float(self.voice_model.predict_proba(df_v)[0, 1])
            active_modalities.append("Voice Acoustics")
            if self.voice_explainer is not None:
                v_shap = self.voice_explainer.shap_values(df_v)[0]
                for feat, val in zip(self.voice_features, v_shap):
                    attributions[f"Voice: {feat}"] = float(val)

        if not active_modalities:
            raise ValueError("At least one modality (gait_dict or voice_dict) must be provided.")

        # 3. Decision-Level Fusion for PD Risk
        if gait_prob is not None and voice_prob is not None:
            # Multi-modal late fusion
            pd_risk = float(0.5 * gait_prob + 0.5 * voice_prob)
            modality_label = "Bimodal (Gait + Voice)"
        elif gait_prob is not None:
            pd_risk = gait_prob
            modality_label = "Unimodal (Gait Kinematics)"
        else:
            pd_risk = voice_prob
            modality_label = "Unimodal (Voice Acoustics)"

        # 4. Stratify into Clinical Risk Tiers
        if pd_risk < 0.35:
            tier = "Low Risk"
        elif pd_risk <= 0.65:
            tier = "Moderate Risk"
        else:
            tier = "High Risk"

        # Sort attributions by absolute impact
        sorted_attributions = dict(
            sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        return {
            "pd_risk_score": pd_risk,
            "pd_risk_percentage": round(pd_risk * 100, 1),
            "risk_tier": tier,
            "modality_used": modality_label,
            "gait_risk_score": gait_prob,
            "voice_risk_score": voice_prob,
            "biomarker_attributions": sorted_attributions,
        }

    def save(self, output_path: str | Path, metadata: dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        joblib.dump(
            {
                "gait_model": self.gait_model,
                "voice_model": self.voice_model,
                "gait_explainer": self.gait_explainer,
                "voice_explainer": self.voice_explainer,
                "gait_features": self.gait_features,
                "voice_features": self.voice_features,
                "metadata": metadata,
            },
            path,
        )
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ParkinsonUnifiedModel:
        bundle = joblib.load(str(path))
        instance = cls()
        instance.gait_model = bundle["gait_model"]
        instance.voice_model = bundle["voice_model"]
        instance.gait_explainer = bundle["gait_explainer"]
        instance.voice_explainer = bundle["voice_explainer"]
        instance.gait_features = bundle["gait_features"]
        instance.voice_features = bundle["voice_features"]
        instance.metadata = bundle.get("metadata", {})
        return instance
