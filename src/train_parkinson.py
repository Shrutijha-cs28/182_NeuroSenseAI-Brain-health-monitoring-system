"""
Training and evaluation script for the NeuroSense AI Parkinson's Disease Module.
Conducts strict subject-level holdout validation across both Gait and Vocal cohorts.
Persists unified model bundle and SHAP explainers to models/neurosense_parkinson.joblib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.oasis_ai.parkinson_model import ParkinsonUnifiedModel


def main():
    parser = argparse.ArgumentParser(description="Train NeuroSense AI Parkinson's Screening Model")
    parser.add_argument("--gait_data", default="data/processed/gait_features.csv", help="Processed gait CSV")
    parser.add_argument("--voice_data", default="data/interim/parkinsons.csv", help="UCI voice CSV")
    parser.add_argument("--output", default="models/neurosense_parkinson.joblib", help="Output model path")
    args = parser.parse_args()

    gait_path = Path(args.gait_data)
    voice_path = Path(args.voice_data)

    if not gait_path.exists():
        raise FileNotFoundError(f"Gait data not found at {gait_path}. Run prepare_gait_data first.")
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice data not found at {voice_path}.")

    model = ParkinsonUnifiedModel()

    # =========================================================
    # 1. Train Gait Biomechanical Stream
    # =========================================================
    df_gait = pd.read_csv(gait_path)
    X_g = df_gait[model.gait_features]
    y_g = df_gait["target"].astype(int)
    groups_g = df_gait["subject_id"]

    gss_gait = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    g_train_idx, g_test_idx = next(gss_gait.split(X_g, y_g, groups=groups_g))

    X_g_train, X_g_test = X_g.iloc[g_train_idx], X_g.iloc[g_test_idx]
    y_g_train, y_g_test = y_g.iloc[g_train_idx], y_g.iloc[g_test_idx]

    model.fit_gait(X_g_train, y_g_train)
    g_probs = model.gait_model.predict_proba(X_g_test)[:, 1]
    g_preds = (g_probs >= 0.5).astype(int)

    g_acc = float(accuracy_score(y_g_test, g_preds))
    g_f1 = float(f1_score(y_g_test, g_preds, average="macro"))
    g_auc = float(roc_auc_score(y_g_test, g_probs))

    # =========================================================
    # 2. Train Vocal Acoustic Stream
    # =========================================================
    df_voice = pd.read_csv(voice_path)
    # Extract subject ID from name (e.g. phon_R01_S01_1 -> S01)
    df_voice["subject_id"] = df_voice["name"].apply(lambda n: str(n).split("_")[2] if "_" in str(n) else str(n))
    X_v = df_voice[model.voice_features]
    y_v = df_voice["status"].astype(int)
    groups_v = df_voice["subject_id"]

    gss_voice = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    v_train_idx, v_test_idx = next(gss_voice.split(X_v, y_v, groups=groups_v))

    X_v_train, X_v_test = X_v.iloc[v_train_idx], X_v.iloc[v_test_idx]
    y_v_train, y_v_test = y_v.iloc[v_train_idx], y_v.iloc[v_test_idx]

    model.fit_voice(X_v_train, y_v_train)
    v_probs = model.voice_model.predict_proba(X_v_test)[:, 1]
    v_preds = (v_probs >= 0.5).astype(int)

    v_acc = float(accuracy_score(y_v_test, v_preds))
    v_f1 = float(f1_score(y_v_test, v_preds, average="macro"))
    v_auc = float(roc_auc_score(y_v_test, v_probs))

    # =========================================================
    # 3. Save Model and Evaluation Metadata
    # =========================================================
    metadata = {
        "model_type": "NeuroSense AI - Parkinson's Screening Engine (Gait + Voice + SHAP)",
        "gait_stream": {
            "dataset": "PhysioNet / devalab/parkinsonsfromgait",
            "accuracy": round(g_acc, 4),
            "macro_f1": round(g_f1, 4),
            "roc_auc": round(g_auc, 4),
            "train_subjects": int(groups_g.iloc[g_train_idx].nunique()),
            "test_subjects": int(groups_g.iloc[g_test_idx].nunique()),
            "features": model.gait_features,
            "default_medians": {f: float(X_g[f].median()) for f in model.gait_features},
        },
        "voice_stream": {
            "dataset": "UCI Parkinson's Disease Phonation Dataset",
            "accuracy": round(v_acc, 4),
            "macro_f1": round(v_f1, 4),
            "roc_auc": round(v_auc, 4),
            "train_subjects": int(groups_v.iloc[v_train_idx].nunique()),
            "test_subjects": int(groups_v.iloc[v_test_idx].nunique()),
            "features": model.voice_features,
            "default_medians": {f: float(X_v[f].median()) for f in model.voice_features},
        },
        "fusion_strategy": "Decision-Level Late Fusion (Calibrated Bimodal Risk Index)",
    }

    model.save(args.output, metadata)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "parkinson_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 60)
    print("NEUROSENSE AI: PARKINSON'S MODEL EVALUATION")
    print("=" * 60)
    print(f"Gait Stream  - Holdout Accuracy: {g_acc:.1%}, ROC-AUC: {g_auc:.3f} ({metadata['gait_stream']['test_subjects']} test subjects)")
    print(f"Voice Stream - Holdout Accuracy: {v_acc:.1%}, ROC-AUC: {v_auc:.3f} ({metadata['voice_stream']['test_subjects']} test subjects)")
    print(f"Saved model to   : {args.output}")
    print(f"Saved metrics to : reports/parkinson_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
