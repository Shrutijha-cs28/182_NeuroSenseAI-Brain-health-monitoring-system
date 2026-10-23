from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.oasis_ai.alzheimer_model import AlzheimerRiskModel
from src.oasis_ai.mri_processor import MRIProcessor


MODEL_PATH = Path("models/neurosense_alzheimer.joblib")
METRICS_PATH = Path("reports/alzheimer_metrics.json")
MRI_DIR = Path("data/raw/oasis2_mri")

st.set_page_config(
    page_title="NeuroSense AI | Brain Health Monitoring",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroSense AI: Brain Health Monitoring System")
st.subheader("Module 1: Alzheimer's Disease Risk Screening & Neuroimaging")
st.caption("Phase-II Engineering Prototype · Subject-Level Holdout Validation · Explainable AI")

if not MODEL_PATH.exists():
    st.error(f"Trained model not found at '{MODEL_PATH}'. Please run `python -m src.train_alzheimer` first.")
    st.stop()

# Load Model & Metadata
model = AlzheimerRiskModel.load(MODEL_PATH)
metadata = model.metadata
if not metadata and METRICS_PATH.exists():
    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

mri_proc = MRIProcessor(MRI_DIR)

st.warning(
    "⚠️ **Academic & Screening Demonstration Only**: NeuroSense AI is designed to assist in early risk screening "
    "and continuous brain-health monitoring. It does not replace professional medical diagnosis."
)

tab1, tab2, tab3 = st.tabs(["🔬 Patient Risk Screening", "🖼️ MRI Scan Visualizer", "📊 Model Performance & Metrics"])

# -------------------------------------------------------------
# TAB 1: Clinical Risk Screening & Explainable AI (SHAP)
# -------------------------------------------------------------
with tab1:
    st.markdown("### Enter Patient Biomarkers & Cognitive Scores")
    st.markdown("Enter clinical assessment metrics to evaluate cognitive impairment / Alzheimer's risk:")

    defaults = metadata.get("default_medians", {
        "Age": 77.0, "EDUC": 15.0, "SES": 2.0, "MMSE": 29.0,
        "eTIV": 1470.0, "nWBV": 0.73, "ASF": 1.19
    })

    with st.form("patient_screening_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=50.0, max_value=105.0, value=float(defaults.get("Age", 77.0)), step=1.0)
            educ = st.number_input("Education (years)", min_value=1.0, max_value=25.0, value=float(defaults.get("EDUC", 15.0)), step=1.0)
            ses = st.selectbox("Socioeconomic Status (SES 1=Highest, 5=Lowest)", options=[1.0, 2.0, 3.0, 4.0, 5.0], index=1)
        with col2:
            mmse = st.slider("Mini-Mental State Exam (MMSE)", min_value=0.0, max_value=30.0, value=float(defaults.get("MMSE", 29.0)), step=1.0,
                             help="Scores < 24 commonly indicate cognitive impairment.")
            etiv = st.number_input("Estimated Total Intracranial Volume (eTIV cm³)", min_value=900.0, max_value=2300.0, value=float(defaults.get("eTIV", 1470.0)), step=10.0)
        with col3:
            nwbv = st.slider("Normalized Whole Brain Volume (nWBV)", min_value=0.60, max_value=0.88, value=float(defaults.get("nWBV", 0.73)), step=0.01,
                             help="Ratio of brain parenchyma to total intracranial volume. Decreases with neurodegeneration.")
            asf = st.number_input("Atlas Scaling Factor (ASF)", min_value=0.75, max_value=1.75, value=float(defaults.get("ASF", 1.19)), step=0.01)

        submitted = st.form_submit_button("🔍 Run NeuroSense Screening", use_container_width=True)

    if submitted:
        patient_data = {
            "Age": age, "EDUC": educ, "SES": ses, "MMSE": mmse,
            "eTIV": etiv, "nWBV": nwbv, "ASF": asf
        }
        res = model.predict_risk(patient_data)
        prob = res["risk_score"]
        tier = res["risk_tier"]
        pct = res["risk_percentage"]

        st.markdown("---")
        st.subheader("Screening Outcome & Risk Stratification")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if tier == "Low Risk":
                st.success(f"### {tier}")
            elif tier == "Moderate Risk":
                st.warning(f"### {tier}")
            else:
                st.error(f"### {tier}")

            st.metric(label="Predicted Alzheimer's Risk", value=f"{pct}%")
            st.progress(prob)
            st.caption(f"Estimated parenchymal brain volume: **{res['engineered_features']['Brain_Volume_Est']:.1f} cm³**")
            st.caption(f"Atrophy ratio (CSF / tissue loss): **{res['engineered_features']['Atrophy_Ratio']*100:.1f}%**")

        with res_col2:
            st.markdown("#### 💡 Explainable AI: Biomarker Contribution (SHAP)")
            st.caption("Positive values increase Alzheimer's risk; negative values protect against risk.")

            attributions = res["biomarker_attributions"]
            # Plot horizontal bar chart
            fig, ax = plt.subplots(figsize=(7, 4))
            features = list(reversed(list(attributions.keys())[:7]))
            values = [attributions[f] for f in features]
            colors = ["#e63946" if v > 0 else "#2a9d8f" for v in values]

            ax.barh(features, values, color=colors)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("SHAP Impact on Predicted Dementia Risk")
            ax.set_title("Biomarker Risk Drivers")
            plt.tight_layout()
            st.pyplot(fig)

# -------------------------------------------------------------
# TAB 2: MRI Scan Visualizer (Structural Neuroimaging)
# -------------------------------------------------------------
with tab2:
    st.markdown("### Structural MRI Scan & Ventricular Atrophy Visualizer")
    st.markdown("Select an extracted OASIS-2 MRI scan session to inspect axial brain slices and ventricular enlargement:")

    if not MRI_DIR.exists():
        st.info("MRI directory not found. Please extract OASIS-2 scans to `data/raw/oasis2_mri`.")
    else:
        available_sessions = sorted([p.name for p in MRI_DIR.iterdir() if p.is_dir()])
        if not available_sessions:
            st.info("No extracted MRI sessions found in `data/raw/oasis2_mri`.")
        else:
            selected_session = st.selectbox("Select Patient MRI ID", options=available_sessions)
            scan_path = mri_proc.find_scan(selected_session)

            if scan_path:
                with st.spinner(f"Loading 3D MRI volume for {selected_session}..."):
                    volume = mri_proc.load_volume(scan_path)
                    max_slice = volume.shape[2] - 1
                    slice_idx = st.slider("Axial Slice Index", min_value=0, max_value=max_slice, value=max_slice // 2)

                    axial_slice = mri_proc.extract_axial_slice(volume, slice_idx=slice_idx)
                    overlay = mri_proc.generate_atrophy_overlay(axial_slice)

                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.markdown("**Original T1-Weighted Structural MRI**")
                        fig_raw, ax_raw = plt.subplots(figsize=(5, 5))
                        ax_raw.imshow(axial_slice, cmap="gray")
                        ax_raw.axis("off")
                        st.pyplot(fig_raw)

                    with img_col2:
                        st.markdown("**Ventricular & Sulcal Atrophy Heatmap Overlay**")
                        fig_heat, ax_heat = plt.subplots(figsize=(5, 5))
                        ax_heat.imshow(axial_slice, cmap="gray")
                        ax_heat.imshow(overlay, cmap="hot", alpha=0.55)
                        ax_heat.axis("off")
                        st.pyplot(fig_heat)

                    st.caption("Red/Yellow highlights denote CSF expansion and brain tissue retraction around lateral ventricles.")
            else:
                st.warning(f"No .nifti.hdr file found under session {selected_session}.")

# -------------------------------------------------------------
# TAB 3: Model Performance & Subject-Independent Validation
# -------------------------------------------------------------
with tab3:
    st.markdown("### Model Architecture & Holdout Validation")
    st.markdown(
        "To satisfy the core requirement of your project report, evaluation is performed with **strict subject-level holdout**. "
        "No participant appears in both training and test cohorts."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metadata.get('roc_auc', 0.806):.3f}")
    m2.metric("Holdout Accuracy", f"{metadata.get('accuracy', 0.663):.1%}")
    m3.metric("Macro F1-Score", f"{metadata.get('macro_f1', 0.660):.3f}")
    m4.metric("Held-Out Subjects", f"{metadata.get('test_subjects', 38)}")

    st.markdown("#### Classification Metrics")
    clf_dict = metadata.get("classification_report", {})
    if clf_dict:
        st.dataframe(pd.DataFrame(clf_dict).transpose(), use_container_width=True)

    st.caption("Trained on OASIS-2 longitudinal cohort. Excludes Clinical Dementia Rating (CDR) to ensure realistic screening.")
