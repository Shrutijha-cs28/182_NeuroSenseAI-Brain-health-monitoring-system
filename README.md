# NeuroSense AI: Brain Health Monitoring System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20SHAP-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NeuroSense AI** is a multimodal AI-assisted brain health monitoring and early risk screening platform designed for **Alzheimer's Disease** and **Parkinson's Disease**. It integrates non-invasive biomarker streams, disease-specific machine learning pipelines, and Explainable AI (SHAP and structural MRI atrophy heatmaps).

> **Disclaimer**: NeuroSense AI is an educational and research prototype intended for risk screening and monitoring demonstrations. It is **not** a diagnostic device and should never replace professional medical judgment.

---

## System Architecture

```
neurosense_ai/
│
├── app.py                             # Interactive Streamlit dashboard
├── requirements.txt                   # Project dependencies
│
├── src/
│   ├── train_alzheimer.py             # Alzheimer's model training pipeline
│   ├── train_parkinson.py             # Parkinson's dual-stream training pipeline
│   │
│   └── oasis_ai/
│       ├── alzheimer_model.py         # XGBoost classifier + SHAP for Alzheimer's
│       ├── mri_processor.py           # 3D NIfTI volume loader & atrophy heatmap visualizer
│       ├── gait_processor.py          # Biomechanical VGRF sensor feature extractor
│       ├── voice_processor.py         # Librosa & Scipy acoustic vocal feature extractor
│       ├── parkinson_model.py         # Bimodal Parkinson's model + SHAP explainability
│       └── prepare_gait_data.py       # Batch feature extractor for gait trials
│
├── data/                              # Local datasets (strictly access-controlled / git-ignored)
│   ├── raw/oasis2_mri/                # Extracted 3D T1 MRI NIfTI volumes
│   ├── raw/gait_force/                # Vertical ground reaction force (VGRF) trials
│   ├── interim/oasis.csv              # OASIS-2 longitudinal clinical table
│   └── interim/parkinsons.csv         # UCI vocal phonation dataset
│
├── models/                            # Trained model bundles (git-ignored)
│   ├── neurosense_alzheimer.joblib
│   └── neurosense_parkinson.joblib
│
└── reports/                           # Evaluation metrics (git-ignored)
    ├── alzheimer_metrics.json
    └── parkinson_metrics.json
```

---

## Implemented Modules

### 🧠 Module 1: Alzheimer's Disease Risk Screening
* **Data**: OASIS-2 Longitudinal Cohort (150 patients, 373 visits) & 596 local NIfTI MRI scans.
* **Feature Engineering**: Normalized Whole Brain Volume (`nWBV`), Estimated Total Intracranial Volume (`eTIV`), Atrophy Ratio ($1 - \text{nWBV}$), and Age-Atrophy Index. `CDR` is excluded to prevent target leakage.
* **Validation**: Strict participant-level holdout (`GroupShuffleSplit`).
* **Performance**: **`80.6%` ROC-AUC**, **`66.3%` Holdout Accuracy** on unseen subjects.
* **Explainability**: Real-time **SHAP biomarker attributions** and **Axial MRI Ventricular Atrophy Heatmaps**.

### 🚶 Module 2: Parkinson's Disease Motor & Vocal Screening
* **Gait Biomechanics**: Vertical ground reaction force (VGRF) dataset (`devalab/parkinsonsfromgait`, 306 trials across 165 participants).
  * Extracted biomarkers: Stride CV (stepping irregularity), Bilateral Gait Asymmetry, Tremor Band Power (3–8 Hz oscillatory power ratio), and Cadence (Hz).
  * Performance: **`71.4%` Holdout Accuracy**, **`0.726` ROC-AUC** across 42 held-out test subjects.
* **Vocal Acoustics**: UCI Parkinson's Phonation Dataset + live `.wav` file analysis using `librosa`.
  * Extracted biomarkers: Pitch stability ($F_0$), Jitter (frequency perturbation), Shimmer (amplitude perturbation), Harmonic-to-Noise Ratio (HNR), and Pitch Period Entropy (PPE).
  * Performance: **`75.5%` Holdout Accuracy**, **`0.673` ROC-AUC** across 8 held-out test subjects.
* **Explainability**: SHAP attribution plots showing how specific motor and vocal features contribute to risk.

---

## Quickstart: Run Locally on Windows

### 1. Set Up Environment
Open PowerShell inside this folder:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train / Verify the Models
To train the Alzheimer's Disease model:
```powershell
python -m src.train_alzheimer --data data/interim/oasis.csv
```

To extract gait features and train the Parkinson's Disease model:
```powershell
python -m src.oasis_ai.prepare_gait_data
python -m src.train_parkinson
```

### 3. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Data Privacy & Governance

In compliance with the **OASIS Brains Data Use Terms** and open medical dataset licensing:
* Raw MRI scans, participant time-series, spreadsheets, and trained `.joblib` binaries remain strictly local on the authorized machine.
* `.gitignore` automatically prevents confidential clinical data from being committed to public GitHub repositories.
