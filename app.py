from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/oasis_prototype.joblib")
st.set_page_config(page_title="NeuroSense AI | Research Prototype", page_icon="🧠", layout="centered")
st.title("🧠 Alzheimer’s NeuroSense AI")
st.caption("OASIS research prototype · visit-level tabular model")

if not MODEL_PATH.exists():
    st.info("Train the prototype first, then reopen this page. See the README for setup steps.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
metadata = bundle["metadata"]
st.warning(
    "Research demonstration only. This model is not a diagnostic or screening device and must not be used "
    "to make care decisions. Its scores reflect a small, single-source dataset and may not generalize."
)

with st.expander("About the prototype evaluation"):
    st.write(f"Subject-level holdout accuracy: **{metadata['accuracy']:.1%}**")
    st.write(f"Subject-level holdout macro F1: **{metadata['macro_f1']:.3f}**")
    st.write(f"Held-out people: **{metadata['test_subjects']}**")
    st.caption("CDR and participant/session identifiers are excluded from model inputs.")

st.subheader("Explore a sample prediction")
st.write("Enter the clinical and demographic fields available in the OASIS table.")
values = {}
with st.form("prediction"):
    cols = st.columns(2)
    for index, feature in enumerate(bundle["columns"]):
        with cols[index % 2]:
            if feature in metadata.get("numeric_columns", []):
                default = float(metadata.get("numeric_defaults", {}).get(feature, 0.0))
                values[feature] = st.number_input(feature, value=default, step=1.0, format="%.2f")
            else:
                choices = metadata.get("categories", {}).get(feature, [])
                values[feature] = st.selectbox(feature, choices) if choices else st.text_input(feature)
    submitted = st.form_submit_button("Run prototype")

if submitted:
    sample = pd.DataFrame([values], columns=bundle["columns"])
    predicted = model.predict(sample)[0]
    st.success(f"Prototype class: **{predicted}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        st.write("Model class scores (not calibrated clinical probabilities):")
        st.dataframe(
            pd.DataFrame({"OASIS group": model.classes_, "Score": probabilities})
            .sort_values("Score", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    st.caption("This output is an educational model estimate, not a diagnosis or medical recommendation.")
