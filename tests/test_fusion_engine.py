"""
Integration test for NeuroSense AI Multimodal Fusion Engine and Longitudinal Tracking.
"""
from src.oasis_ai.fusion_engine import NeuroSenseFusionEngine


def run_test():
    engine = NeuroSenseFusionEngine()
    print("Alzheimer Model Loaded:", engine.alz_model is not None)
    print("Parkinson Model Loaded:", engine.pd_model is not None)

    # 1. Visit 1 (Healthy / Low Risk)
    sample_alz_v1 = {
        "Age": 74.0, "EDUC": 14.0, "SES": 2.0, "MMSE": 28.0,
        "eTIV": 1510.0, "nWBV": 0.74, "ASF": 1.16
    }
    sample_gait_v1 = {
        "mean_force": 2.4, "force_std": 0.18, "stride_cv": 0.08,
        "gait_asymmetry": 0.82, "cadence_hz": 1.6, "tremor_band_power": 0.22,
        "left_right_ratio": 0.96, "force_peak_to_mean": 1.4
    }

    res1 = engine.evaluate_patient("PATIENT_001", alzheimer_inputs=sample_alz_v1, gait_inputs=sample_gait_v1)
    print(f"Visit 1 -> BHI: {res1['bhi_score']} ({res1['bhi_status']}) | AD Risk: {res1['ad_risk_pct']}% | PD Risk: {res1['pd_risk_pct']}%")

    # 2. Visit 2: 12 months later (Significant cognitive drop, motor stable)
    sample_alz_v2 = {
        "Age": 75.0, "EDUC": 14.0, "SES": 2.0, "MMSE": 21.0,
        "eTIV": 1510.0, "nWBV": 0.67, "ASF": 1.16
    }
    res2 = engine.evaluate_patient("PATIENT_001", alzheimer_inputs=sample_alz_v2, gait_inputs=sample_gait_v1)
    decline = res2["longitudinal_decline"]
    print(f"Visit 2 -> BHI: {res2['bhi_score']} ({res2['bhi_status']}) | Delta BHI: {decline['delta_bhi']}")
    print(f"Longitudinal Trend: {decline['trend']} | Alert Triggered: {decline['alert']}")
    print("Top 3 Cross-Condition Biomarkers (SHAP):", list(res2["combined_biomarkers"].items())[:3])


if __name__ == "__main__":
    run_test()
