"""
Unified Multimodal Fusion Engine for NeuroSense AI.
Combines Alzheimer's Disease and Parkinson's Disease risk streams
into the composite Brain Health Indicator (BHI: 0 - 100) and orchestrates
cross-disease Explainable AI and longitudinal tracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from src.oasis_ai.alzheimer_model import AlzheimerRiskModel
from src.oasis_ai.parkinson_model import ParkinsonUnifiedModel
from src.oasis_ai.longitudinal_db import LongitudinalDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class NeuroSenseFusionEngine:
    def __init__(
        self,
        alz_model_path: str | Path | None = None,
        pd_model_path: str | Path | None = None,
        db_path: str | Path | None = None
    ):
        self.alz_model_path = Path(alz_model_path) if alz_model_path else PROJECT_ROOT / "models" / "neurosense_alzheimer.joblib"
        self.pd_model_path = Path(pd_model_path) if pd_model_path else PROJECT_ROOT / "models" / "neurosense_parkinson.joblib"
        db_file = Path(db_path) if db_path else PROJECT_ROOT / "database" / "patient_history.db"
        self.alz_model: AlzheimerRiskModel | None = None
        self.pd_model: ParkinsonUnifiedModel | None = None
        self.db = LongitudinalDatabase(db_file)

        if self.alz_model_path.exists():
            self.alz_model = AlzheimerRiskModel.load(self.alz_model_path)
        if self.pd_model_path.exists():
            self.pd_model = ParkinsonUnifiedModel.load(self.pd_model_path)

    def compute_bhi(self, ad_risk: float | None, pd_risk: float | None) -> tuple[float, str]:
        """
        Calculates the composite Brain Health Indicator (BHI: 0 - 100).
        100 represents optimal brain health, while lower scores reflect higher neurodegenerative risk.
        Dominant-risk weighted formulation handles single or dual condition risk.
        """
        active_risks = [r for r in [ad_risk, pd_risk] if r is not None]
        if not active_risks:
            return 100.0, "Optimal (No Risk Detected)"

        # Dominant risk has 65% weight, mean has 35% weight
        max_r = max(active_risks)
        mean_r = sum(active_risks) / len(active_risks)
        composite_risk = (max_r * 0.65) + (mean_r * 0.35)

        bhi = round(float(np.clip((1.0 - composite_risk) * 100, 0.0, 100.0)), 1)

        # Risk tiering
        if bhi >= 75.0:
            status = "Optimal / Low Risk"
        elif bhi >= 50.0:
            status = "Moderate Risk (Monitoring Recommended)"
        else:
            status = "High Risk (Clinical Referral Recommended)"

        return bhi, status

    def evaluate_patient(
        self,
        patient_id: str,
        alzheimer_inputs: dict[str, float] | None = None,
        gait_inputs: dict[str, float] | None = None,
        voice_inputs: dict[str, float] | None = None,
        record_history: bool = True
    ) -> dict[str, Any]:
        """
        Conducts a unified multimodal screening assessment for a patient.
        Gracefully handles asymmetric or missing modalities.
        """
        ad_results: dict[str, Any] | None = None
        pd_results: dict[str, Any] | None = None
        ad_risk: float | None = None
        pd_risk: float | None = None
        combined_attributions: dict[str, float] = {}

        # 1. Run Alzheimer's Module
        if alzheimer_inputs is not None and self.alz_model is not None:
            ad_results = self.alz_model.predict_risk(alzheimer_inputs)
            ad_risk = ad_results["risk_score"]
            for feat, val in ad_results["biomarker_attributions"].items():
                combined_attributions[f"Alzheimer's: {feat}"] = val

        # 2. Run Parkinson's Module
        if (gait_inputs is not None or voice_inputs is not None) and self.pd_model is not None:
            pd_results = self.pd_model.predict_risk(gait_dict=gait_inputs, voice_dict=voice_inputs)
            pd_risk = pd_results["pd_risk_score"]
            for feat, val in pd_results["biomarker_attributions"].items():
                combined_attributions[f"Parkinson's: {feat}"] = val

        # 3. Compute Composite Brain Health Indicator (BHI)
        bhi_score, bhi_status = self.compute_bhi(ad_risk, pd_risk)

        # Sort all biomarkers across conditions by absolute severity
        sorted_attributions = dict(
            sorted(combined_attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        # 4. Record to Longitudinal Database
        visit_number = 1
        decline_metrics: dict[str, Any] = {}
        if record_history and patient_id:
            visit_number = self.db.add_assessment(
                patient_id=patient_id,
                ad_risk=ad_risk,
                pd_risk=pd_risk,
                bhi_score=bhi_score,
                risk_tier=bhi_status,
                biomarkers=sorted_attributions
            )
            decline_metrics = self.db.compute_decline_velocity(patient_id)

        return {
            "patient_id": patient_id,
            "visit_number": visit_number,
            "bhi_score": bhi_score,
            "bhi_status": bhi_status,
            "ad_risk": ad_risk,
            "ad_risk_pct": round(ad_risk * 100, 1) if ad_risk is not None else None,
            "pd_risk": pd_risk,
            "pd_risk_pct": round(pd_risk * 100, 1) if pd_risk is not None else None,
            "ad_details": ad_results,
            "pd_details": pd_results,
            "combined_biomarkers": sorted_attributions,
            "longitudinal_decline": decline_metrics,
        }
