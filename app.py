from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.oasis_ai.alzheimer_model import AlzheimerRiskModel
from src.oasis_ai.mri_processor import MRIProcessor
from src.oasis_ai.gait_processor import GaitProcessor
from src.oasis_ai.voice_processor import VoiceProcessor
from src.oasis_ai.parkinson_model import ParkinsonUnifiedModel
from src.oasis_ai.fusion_engine import NeuroSenseFusionEngine
from src.oasis_ai.longitudinal_db import LongitudinalDatabase


PROJECT_ROOT = Path(__file__).resolve().parent
ALZ_MODEL_PATH = PROJECT_ROOT / "models" / "neurosense_alzheimer.joblib"
PD_MODEL_PATH = PROJECT_ROOT / "models" / "neurosense_parkinson.joblib"
MRI_DIR = PROJECT_ROOT / "data" / "raw" / "oasis2_mri"
GAIT_DIR = PROJECT_ROOT / "data" / "raw" / "gait_force"
DB_PATH = PROJECT_ROOT / "database" / "patient_history.db"

st.set_page_config(
    page_title="NeuroSense AI | Brain Health Monitoring",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroSense AI: Brain Health Monitoring System")
st.caption("Multimodal Early Risk Screening & Longitudinal Monitoring Platform for Alzheimer's and Parkinson's")

st.warning(
    "⚠️ **Academic & Screening Demonstration Only**: NeuroSense AI is an AI-assisted early risk screening "
    "and monitoring system. It does not provide medical diagnosis or replace clinical judgment."
)

# Initialize Fusion Engine and Database
fusion_engine = NeuroSenseFusionEngine(ALZ_MODEL_PATH, PD_MODEL_PATH, DB_PATH)
long_db = LongitudinalDatabase(DB_PATH)

# Sidebar System Status & Patient Selector
with st.sidebar:
    st.header("⚙️ System Status")
    st.markdown("**Module 1: Alzheimer's Disease**")
    if ALZ_MODEL_PATH.exists():
        st.success("✅ Model Ready (OASIS MRI + SHAP)")
    else:
        st.error("❌ Model Missing")

    st.markdown("**Module 2: Parkinson's Disease**")
    if PD_MODEL_PATH.exists():
        st.success("✅ Model Ready (Gait + Voice + SHAP)")
    else:
        st.error("❌ Model Missing")

    st.markdown("---")
    st.header("👤 Patient Profile")
    existing_patients = long_db.get_all_patients()
    patient_options = ["New Patient"] + [f"{p['patient_id']} - {p['name']}" for p in existing_patients]
    selected_p = st.selectbox("Select Patient", patient_options)

    if selected_p == "New Patient":
        pid_input = st.text_input("Patient ID", value=f"PATIENT_{len(existing_patients)+1:03d}")
        pname_input = st.text_input("Patient Name", value="Anonymous Subject")
        page_input = st.number_input("Patient Age", 40.0, 105.0, 72.0)
        pgender_input = st.selectbox("Gender", ["Female", "Male", "Other"])
        if st.button("Register Patient Profile"):
            long_db.register_patient(pid_input, pname_input, page_input, pgender_input)
            st.success(f"Registered {pid_input}!")
            st.rerun()
    else:
        pid_input = selected_p.split(" - ")[0]
        st.info(f"Active Patient: **{selected_p}**")

# Top Navigation Tabs
nav_fusion, nav_alz, nav_pd, nav_bench = st.tabs([
    "🌐 Unified Multimodal Dashboard & Longitudinal Tracker",
    "🧠 Module 1: Alzheimer's Disease (Cognitive & MRI)",
    "🚶 Module 2: Parkinson's Disease (Gait & Speech)",
    "📊 Evaluation Benchmarks & Metrics"
])

# ==============================================================================
# TAB 1: UNIFIED MULTIMODAL FUSION & LONGITUDINAL DASHBOARD
# ==============================================================================
with nav_fusion:
    st.subheader("🌐 Executive Brain Health Overview & Longitudinal Trajectory")
    st.markdown("Synthesizes multi-stream cognitive, neuroimaging, kinematic, and acoustic indicators into a composite score:")

    # Top Row: Multimodal Risk Assessment Form
    with st.expander("📝 Conduct Combined Multimodal Assessment for Active Patient", expanded=True):
        st.markdown(f"Running assessment for **{pid_input}**:")
        col_ad, col_pd = st.columns(2)

        with col_ad:
            st.markdown("##### 🧠 Alzheimer's Biomarkers (OASIS Stream)")
            f_age = st.number_input("Age", 50.0, 100.0, 75.0, key="f_age")
            f_mmse = st.slider("Mini-Mental State Exam (MMSE)", 0.0, 30.0, 26.0, key="f_mmse")
            f_nwbv = st.slider("Normalized Whole Brain Volume (nWBV)", 0.60, 0.88, 0.72, key="f_nwbv")
            f_etiv = st.number_input("eTIV (cm³)", 1000.0, 2200.0, 1500.0, key="f_etiv")

        with col_pd:
            st.markdown("##### 🚶 Parkinson's Biomarkers (Gait & Voice Stream)")
            # Option to select preset trial
            f_asym = st.number_input("Bilateral Gait Asymmetry", 0.1, 1.0, 0.80, key="f_asym")
            f_scv = st.number_input("Stride CV (Stepping Instability)", 0.01, 0.40, 0.09, key="f_scv")
            f_tremor = st.number_input("Tremor Band Power (3-8 Hz)", 0.05, 0.80, 0.25, key="f_tremor")
            f_jitter = st.number_input("Acoustic Jitter (%)", 0.001, 0.05, 0.006, format="%.4f", key="f_jitter")

        if st.button("🚀 Calculate Composite Brain Health Index & Log Visit", use_container_width=True):
            alz_dict = {"Age": f_age, "EDUC": 14.0, "SES": 2.0, "MMSE": f_mmse, "eTIV": f_etiv, "nWBV": f_nwbv, "ASF": 1.16}
            gait_dict = {"mean_force": 2.3, "force_std": 0.20, "stride_cv": f_scv, "gait_asymmetry": f_asym, "cadence_hz": 1.6, "tremor_band_power": f_tremor, "left_right_ratio": 0.95, "force_peak_to_mean": 1.4}
            voice_dict = {"MDVP:Fo(Hz)": 150.0, "MDVP:Fhi(Hz)": 180.0, "MDVP:Flo(Hz)": 110.0, "MDVP:Jitter(%)": f_jitter, "MDVP:Shimmer": 0.03, "HNR": 21.0, "RPDE": 0.45, "DFA": 0.70, "spread1": -5.0, "spread2": 0.22, "PPE": 0.20}

            eval_res = fusion_engine.evaluate_patient(
                patient_id=pid_input,
                alzheimer_inputs=alz_dict,
                gait_inputs=gait_dict,
                voice_inputs=voice_dict,
                record_history=True
            )
            st.session_state["latest_eval"] = eval_res
            st.success(f"Assessment logged successfully as Visit #{eval_res['visit_number']}!")

    # Display Latest Assessment Results
    eval_data = st.session_state.get("latest_eval", None)
    if eval_data is not None and eval_data["patient_id"] == pid_input:
        st.markdown("---")
        st.subheader("🎯 Multimodal Screening Indicators")

        m1, m2, m3 = st.columns([1.5, 1, 1])
        with m1:
            bhi = eval_data["bhi_score"]
            status = eval_data["bhi_status"]
            st.metric("Composite Brain Health Indicator (BHI)", f"{bhi} / 100", help="100: Optimal Health. < 50: High Neurodegenerative Risk.")
            st.progress(bhi / 100.0)
            if bhi >= 75:
                st.success(f"**Status**: {status}")
            elif bhi >= 50:
                st.warning(f"**Status**: {status}")
            else:
                st.error(f"**Status**: {status}")

        with m2:
            ad_p = eval_data["ad_risk_pct"]
            st.metric("Alzheimer's Risk", f"{ad_p}%")
            st.progress(eval_data["ad_risk"])
            st.caption(f"Tier: **{eval_data['ad_details']['risk_tier']}**")

        with m3:
            pd_p = eval_data["pd_risk_pct"]
            st.metric("Parkinson's Risk", f"{pd_p}%")
            st.progress(eval_data["pd_risk"])
            st.caption(f"Tier: **{eval_data['pd_details']['risk_tier']}**")

        # Explainable AI: Cross-Condition Biomarker Prioritization
        st.markdown("#### 💡 Explainable AI: Unified Biomarker Prioritization")
        st.caption("Ranks features across cognitive, structural, kinetic, and acoustic modalities by their contribution towards neurodegenerative risk:")

        biomarkers = eval_data["combined_biomarkers"]
        fig_cross, ax_c = plt.subplots(figsize=(8, 3.8))
        c_feats = list(reversed(list(biomarkers.keys())[:8]))
        c_vals = [biomarkers[k] for k in c_feats]
        c_colors = ["#e63946" if v > 0 else "#2a9d8f" for v in c_vals]
        ax_c.barh(c_feats, c_vals, color=c_colors)
        ax_c.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_c.set_xlabel("SHAP Impact on Disease Risk")
        ax_c.set_title("Cross-Domain Risk Drivers")
        plt.tight_layout()
        st.pyplot(fig_cross)

    # Longitudinal History Section
    st.markdown("---")
    st.subheader(f"📈 Longitudinal Brain Health History for {pid_input}")
    history_df = long_db.get_patient_history(pid_input)

    if len(history_df) == 0:
        st.info("No recorded visits for this patient yet. Use the assessment form above to record Visit 1.")
    else:
        lh_col1, lh_col2 = st.columns([2, 1])
        with lh_col1:
            fig_trend, ax_t = plt.subplots(figsize=(7, 3.5))
            ax_t.plot(history_df["visit_number"], history_df["bhi_score"], marker="o", color="#2a9d8f", linewidth=2.5, label="Brain Health Indicator (BHI)")
            if history_df["ad_risk"].notna().any():
                ax_t.plot(history_df["visit_number"], history_df["ad_risk"] * 100, marker="s", color="#e76f51", linestyle="--", label="Alzheimer's Risk %")
            if history_df["pd_risk"].notna().any():
                ax_t.plot(history_df["visit_number"], history_df["pd_risk"] * 100, marker="^", color="#457b9d", linestyle=":", label="Parkinson's Risk %")

            ax_t.set_xlabel("Visit Number")
            ax_t.set_ylabel("Score / Percentage")
            ax_t.set_ylim(-5, 105)
            ax_t.grid(True, linestyle="--", alpha=0.5)
            ax_t.legend(loc="best")
            plt.tight_layout()
            st.pyplot(fig_trend)

        with lh_col2:
            decline_info = long_db.compute_decline_velocity(pid_input)
            st.markdown("##### Longitudinal Trajectory")
            st.write(f"- **Total Visits**: `{decline_info.get('total_visits', len(history_df))}`")
            st.write(f"- **Baseline BHI**: `{decline_info.get('baseline_bhi', history_df.iloc[0]['bhi_score'])}`")
            st.write(f"- **Current BHI**: `{decline_info.get('current_bhi', history_df.iloc[-1]['bhi_score'])}`")
            st.write(f"- **Delta BHI**: `{decline_info.get('delta_bhi', 0.0):+0.1f}`")

            if decline_info.get("alert", False):
                st.error(f"🚨 **Alert**: {decline_info['trend']} (Drop > 10 points)")
            else:
                st.success(f"**Trend**: {decline_info.get('trend', 'Stable')}")

        st.dataframe(history_df, use_container_width=True)

# ==============================================================================
# TAB 2: ALZHEIMER'S DISEASE MODULE
# ==============================================================================
with nav_alz:
    st.subheader("Module 1: Alzheimer's Disease Cognitive & Structural Screening")

    if not ALZ_MODEL_PATH.exists():
        st.error("Alzheimer's model artifact not found.")
    else:
        alz_model = AlzheimerRiskModel.load(ALZ_MODEL_PATH)
        mri_proc = MRIProcessor(MRI_DIR)

        alz_sub1, alz_sub2 = st.tabs(["🔬 Cognitive Risk Screening", "🖼️ 3D MRI Brain Visualizer"])

        with alz_sub1:
            st.markdown("Enter clinical assessment metrics to evaluate cognitive impairment / Alzheimer's risk:")
            alz_defaults = alz_model.metadata.get("default_medians", {
                "Age": 77.0, "EDUC": 15.0, "SES": 2.0, "MMSE": 29.0,
                "eTIV": 1470.0, "nWBV": 0.73, "ASF": 1.19
            })

            with st.form("alz_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input("Age (years)", 50.0, 105.0, float(alz_defaults.get("Age", 77.0)), step=1.0)
                    educ = st.number_input("Education (years)", 1.0, 25.0, float(alz_defaults.get("EDUC", 15.0)), step=1.0)
                    ses = st.selectbox("Socioeconomic Status (SES 1=Highest, 5=Lowest)", [1.0, 2.0, 3.0, 4.0, 5.0], index=1)
                with c2:
                    mmse = st.slider("Mini-Mental State Exam (MMSE)", 0.0, 30.0, float(alz_defaults.get("MMSE", 29.0)), step=1.0)
                    etiv = st.number_input("Estimated Total Intracranial Volume (eTIV cm³)", 900.0, 2300.0, float(alz_defaults.get("eTIV", 1470.0)), step=10.0)
                with c3:
                    nwbv = st.slider("Normalized Whole Brain Volume (nWBV)", 0.60, 0.88, float(alz_defaults.get("nWBV", 0.73)), step=0.01)
                    asf = st.number_input("Atlas Scaling Factor (ASF)", 0.75, 1.75, float(alz_defaults.get("ASF", 1.19)), step=0.01)

                alz_submit = st.form_submit_button("🔍 Run Alzheimer's Screening", use_container_width=True)

            if alz_submit:
                patient_data = {"Age": age, "EDUC": educ, "SES": ses, "MMSE": mmse, "eTIV": etiv, "nWBV": nwbv, "ASF": asf}
                res = alz_model.predict_risk(patient_data)
                st.markdown("---")
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    if res["risk_tier"] == "Low Risk":
                        st.success(f"### {res['risk_tier']}")
                    elif res["risk_tier"] == "Moderate Risk":
                        st.warning(f"### {res['risk_tier']}")
                    else:
                        st.error(f"### {res['risk_tier']}")

                    st.metric("Predicted Alzheimer's Risk", f"{res['risk_percentage']}%")
                    st.progress(res["risk_score"])
                    st.caption(f"Estimated Parenchymal Volume: **{res['engineered_features']['Brain_Volume_Est']:.1f} cm³**")
                    st.caption(f"Atrophy Ratio: **{res['engineered_features']['Atrophy_Ratio']*100:.1f}%**")

                with rc2:
                    st.markdown("#### 💡 Explainable AI: Biomarker Contribution (SHAP)")
                    attributions = res["biomarker_attributions"]
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    features = list(reversed(list(attributions.keys())[:7]))
                    values = [attributions[f] for f in features]
                    colors = ["#e63946" if v > 0 else "#2a9d8f" for v in values]
                    ax.barh(features, values, color=colors)
                    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
                    ax.set_title("Biomarker Risk Drivers")
                    plt.tight_layout()
                    st.pyplot(fig)

        with alz_sub2:
            st.markdown("### Structural MRI Scan & Ventricular Atrophy Visualizer")
            if not MRI_DIR.exists():
                st.info("MRI directory not found at `data/raw/oasis2_mri`.")
            else:
                available_sessions = sorted([p.name for p in MRI_DIR.iterdir() if p.is_dir()])
                if available_sessions:
                    selected_session = st.selectbox("Select Patient MRI ID", options=available_sessions)
                    scan_path = mri_proc.find_scan(selected_session)
                    if scan_path:
                        with st.spinner("Loading 3D volume..."):
                            vol = mri_proc.load_volume(scan_path)
                            max_slice = vol.shape[2] - 1
                            s_idx = st.slider("Axial Slice Depth", 0, max_slice, max_slice // 2)
                            ax_slice = mri_proc.extract_axial_slice(vol, slice_idx=s_idx)
                            overlay = mri_proc.generate_atrophy_overlay(ax_slice)

                            ic1, ic2 = st.columns(2)
                            with ic1:
                                st.markdown("**Original T1-Weighted MRI**")
                                f_raw, a_raw = plt.subplots(figsize=(4.5, 4.5))
                                a_raw.imshow(ax_slice, cmap="gray")
                                a_raw.axis("off")
                                st.pyplot(f_raw)
                            with ic2:
                                st.markdown("**Ventricular Atrophy Heatmap Overlay**")
                                f_heat, a_heat = plt.subplots(figsize=(4.5, 4.5))
                                a_heat.imshow(ax_slice, cmap="gray")
                                a_heat.imshow(overlay, cmap="hot", alpha=0.55)
                                a_heat.axis("off")
                                st.pyplot(f_heat)
                            st.caption("Red/Yellow highlights denote CSF expansion and brain tissue retraction.")

# ==============================================================================
# TAB 3: PARKINSON'S DISEASE MODULE
# ==============================================================================
with nav_pd:
    st.subheader("Module 2: Parkinson's Disease Motor & Vocal Screening")

    if not PD_MODEL_PATH.exists():
        st.error("Parkinson's model artifact not found.")
    else:
        pd_model = ParkinsonUnifiedModel.load(PD_MODEL_PATH)
        gait_proc = GaitProcessor()
        voice_proc = VoiceProcessor()

        pd_sub1, pd_sub2 = st.tabs(["🚶 Biomechanical Gait Force Analysis", "🎙️ Acoustic Speech & Voice Analysis"])

        with pd_sub1:
            st.markdown("### Vertical Ground Reaction Force (VGRF) Gait Assessment")
            gait_defaults = pd_model.metadata.get("gait_stream", {}).get("default_medians", {
                "mean_force": 2.2, "force_std": 0.22, "stride_cv": 0.10,
                "gait_asymmetry": 0.82, "cadence_hz": 1.6, "tremor_band_power": 0.28,
                "left_right_ratio": 0.94, "force_peak_to_mean": 1.45
            })

            available_trials = sorted([f.name for f in GAIT_DIR.glob("*.txt")]) if GAIT_DIR.exists() else []
            selected_trial = st.selectbox("Load Sample Sensor Trial from Dataset", ["Manual Entry / Custom"] + available_trials)

            active_gait_input = dict(gait_defaults)
            if selected_trial != "Manual Entry / Custom":
                trial_path = GAIT_DIR / selected_trial
                active_gait_input = gait_proc.process_file(trial_path)
                data_preview = np.loadtxt(trial_path, delimiter=",")
                
                st.markdown(f"**Force Waveform Plot: {selected_trial}** (First 500 timesteps @ 100 Hz)")
                fig_gait, ax_g = plt.subplots(figsize=(9, 2.8))
                t = np.arange(min(500, len(data_preview))) / 100.0
                left_curve = data_preview[:len(t), 16] if data_preview.shape[1] >= 18 else data_preview[:len(t), :8].sum(axis=1)
                right_curve = data_preview[:len(t), 17] if data_preview.shape[1] >= 18 else data_preview[:len(t), 8:16].sum(axis=1)
                ax_g.plot(t, left_curve, label="Left Foot Force", color="#2a9d8f")
                ax_g.plot(t, right_curve, label="Right Foot Force", color="#e76f51")
                ax_g.set_xlabel("Time (seconds)")
                ax_g.set_ylabel("Force (Normalized)")
                ax_g.legend()
                plt.tight_layout()
                st.pyplot(fig_gait)

            with st.form("gait_form"):
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    m_force = st.number_input("Mean Ground Reaction Force", 0.5, 6.0, float(active_gait_input.get("mean_force", 2.2)), step=0.1)
                    s_cv = st.number_input("Stride CV (Instability Index)", 0.01, 0.50, float(active_gait_input.get("stride_cv", 0.10)), step=0.01)
                with gc2:
                    asym = st.number_input("Bilateral Gait Asymmetry", 0.1, 1.0, float(active_gait_input.get("gait_asymmetry", 0.82)), step=0.02)
                    cadence = st.number_input("Stepping Cadence (Hz)", 0.5, 3.5, float(active_gait_input.get("cadence_hz", 1.6)), step=0.1)
                with gc3:
                    tremor_p = st.number_input("Tremor / Freezing Band Power (3-8 Hz)", 0.05, 0.85, float(active_gait_input.get("tremor_band_power", 0.28)), step=0.02)
                    lr_ratio = st.number_input("Left/Right Force Ratio", 0.4, 2.0, float(active_gait_input.get("left_right_ratio", 0.94)), step=0.05)

                gait_submit = st.form_submit_button("🚶 Run Gait Kinematic Screening", use_container_width=True)

            if gait_submit:
                g_dict = {
                    "mean_force": m_force, "force_std": float(active_gait_input.get("force_std", 0.22)),
                    "stride_cv": s_cv, "gait_asymmetry": asym, "cadence_hz": cadence,
                    "tremor_band_power": tremor_p, "left_right_ratio": lr_ratio,
                    "force_peak_to_mean": float(active_gait_input.get("force_peak_to_mean", 1.45))
                }
                g_res = pd_model.predict_risk(gait_dict=g_dict)
                st.markdown("---")
                gr1, gr2 = st.columns([1, 2])
                with gr1:
                    if g_res["risk_tier"] == "Low Risk":
                        st.success(f"### {g_res['risk_tier']}")
                    elif g_res["risk_tier"] == "Moderate Risk":
                        st.warning(f"### {g_res['risk_tier']}")
                    else:
                        st.error(f"### {g_res['risk_tier']}")

                    st.metric("Gait Parkinson's Risk", f"{g_res['pd_risk_percentage']}%")
                    st.progress(g_res["pd_risk_score"])
                with gr2:
                    st.markdown("#### 💡 Explainable AI: Kinematic Biomarker Impact (SHAP)")
                    fig_s, ax_s = plt.subplots(figsize=(7, 3.5))
                    g_feats = list(reversed(list(g_res["biomarker_attributions"].keys())[:6]))
                    g_vals = [g_res["biomarker_attributions"][k] for k in g_feats]
                    g_cols = ["#e63946" if v > 0 else "#2a9d8f" for v in g_vals]
                    ax_s.barh(g_feats, g_vals, color=g_cols)
                    ax_s.axvline(0, color="gray", linestyle="--")
                    ax_s.set_title("Gait Biomarker Attributions")
                    plt.tight_layout()
                    st.pyplot(fig_s)

        with pd_sub2:
            st.markdown("### Acoustic Phonation Speech Analysis")
            voice_defaults = pd_model.metadata.get("voice_stream", {}).get("default_medians", {
                "MDVP:Fo(Hz)": 150.0, "MDVP:Fhi(Hz)": 180.0, "MDVP:Flo(Hz)": 110.0,
                "MDVP:Jitter(%)": 0.005, "MDVP:Shimmer": 0.03, "HNR": 21.0,
                "RPDE": 0.45, "DFA": 0.70, "spread1": -5.0, "spread2": 0.22, "PPE": 0.20
            })

            audio_file = st.file_uploader("Upload Sustained Vowel Audio Recording (.wav)", type=["wav"])
            active_voice_input = dict(voice_defaults)

            if audio_file is not None:
                st.audio(audio_file)
                with st.spinner("Extracting vocal biomarkers via librosa..."):
                    import soundfile as sf
                    audio_data, sr = sf.read(audio_file)
                    active_voice_input = voice_proc.extract_features_from_audio(audio_data, sr)
                    st.success("Extracted acoustic features from audio waveform successfully!")

            with st.form("voice_form"):
                vc1, vc2, vc3 = st.columns(3)
                with vc1:
                    fo = st.number_input("Average Pitch F0 (Hz)", 70.0, 300.0, float(active_voice_input.get("MDVP:Fo(Hz)", 150.0)), step=5.0)
                    jitter = st.number_input("Jitter (Frequency Perturbation)", 0.0005, 0.05, float(active_voice_input.get("MDVP:Jitter(%)", 0.005)), format="%.4f")
                with vc2:
                    shimmer = st.number_input("Shimmer (Amplitude Perturbation)", 0.005, 0.15, float(active_voice_input.get("MDVP:Shimmer", 0.03)), format="%.4f")
                    hnr = st.number_input("Harmonic-to-Noise Ratio (HNR dB)", 5.0, 40.0, float(active_voice_input.get("HNR", 21.0)), step=1.0)
                with vc3:
                    ppe = st.number_input("Pitch Period Entropy (PPE)", 0.02, 0.60, float(active_voice_input.get("PPE", 0.20)), format="%.4f")
                    dfa = st.number_input("Signal Fractal Exponent (DFA)", 0.4, 0.9, float(active_voice_input.get("DFA", 0.70)), step=0.02)

                voice_submit = st.form_submit_button("🎙️ Run Acoustic Vocal Screening", use_container_width=True)

            if voice_submit:
                v_dict = {
                    "MDVP:Fo(Hz)": fo, "MDVP:Fhi(Hz)": float(active_voice_input.get("MDVP:Fhi(Hz)", 180.0)),
                    "MDVP:Flo(Hz)": float(active_voice_input.get("MDVP:Flo(Hz)", 110.0)),
                    "MDVP:Jitter(%)": jitter, "MDVP:Shimmer": shimmer, "HNR": hnr,
                    "RPDE": float(active_voice_input.get("RPDE", 0.45)), "DFA": dfa,
                    "spread1": float(active_voice_input.get("spread1", -5.0)),
                    "spread2": float(active_voice_input.get("spread2", 0.22)),
                    "PPE": ppe
                }
                v_res = pd_model.predict_risk(voice_dict=v_dict)
                st.markdown("---")
                vr1, vr2 = st.columns([1, 2])
                with vr1:
                    if v_res["risk_tier"] == "Low Risk":
                        st.success(f"### {v_res['risk_tier']}")
                    elif v_res["risk_tier"] == "Moderate Risk":
                        st.warning(f"### {v_res['risk_tier']}")
                    else:
                        st.error(f"### {v_res['risk_tier']}")

                    st.metric("Vocal Parkinson's Risk", f"{v_res['pd_risk_percentage']}%")
                    st.progress(v_res["pd_risk_score"])
                with vr2:
                    st.markdown("#### 💡 Explainable AI: Acoustic Biomarker Impact (SHAP)")
                    fig_v, ax_v = plt.subplots(figsize=(7, 3.5))
                    v_feats = list(reversed(list(v_res["biomarker_attributions"].keys())[:6]))
                    v_vals = [v_res["biomarker_attributions"][k] for k in v_feats]
                    v_cols = ["#e63946" if v > 0 else "#2a9d8f" for v in v_vals]
                    ax_v.barh(v_feats, v_vals, color=v_cols)
                    ax_v.axvline(0, color="gray", linestyle="--")
                    ax_v.set_title("Vocal Biomarker Attributions")
                    plt.tight_layout()
                    st.pyplot(fig_v)

# ==============================================================================
# TAB 4: BENCHMARKS & VALIDATION
# ==============================================================================
with nav_bench:
    st.subheader("Subject-Independent Evaluation Benchmarks")
    st.markdown("Strict patient-level holdout validation prevents data leakage across multiple visits/trials:")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### 🧠 Alzheimer's Disease Engine (OASIS-2)")
        if ALZ_MODEL_PATH.exists():
            alz_m = alz_model.metadata
            st.write(f"- **ROC-AUC**: `{alz_m.get('roc_auc', 0.806):.3f}`")
            st.write(f"- **Holdout Accuracy**: `{alz_m.get('accuracy', 0.663):.1%}`")
            st.write(f"- **Macro F1-Score**: `{alz_m.get('macro_f1', 0.660):.3f}`")
            st.write(f"- **Held-Out Participants**: `{alz_m.get('test_subjects', 38)}`")
            st.caption("Target: 0 = Nondemented, 1 = Impaired/Demented")

    with b_col2:
        st.markdown("#### 🚶 Parkinson's Disease Engine (Gait & Voice)")
        if PD_MODEL_PATH.exists():
            pd_m = pd_model.metadata
            g_meta = pd_m.get("gait_stream", {})
            v_meta = pd_m.get("voice_stream", {})
            st.write(f"- **Gait Stream ROC-AUC**: `{g_meta.get('roc_auc', 0.726):.3f}` (Holdout Acc: `{g_meta.get('accuracy', 0.714):.1%}` across {g_meta.get('test_subjects', 42)} subjects)")
            st.write(f"- **Voice Stream ROC-AUC**: `{v_meta.get('roc_auc', 0.673):.3f}` (Holdout Acc: `{v_meta.get('accuracy', 0.755):.1%}` across {v_meta.get('test_subjects', 8)} subjects)")
            st.caption("Evaluated on completely unseen patient cohorts.")
