"""
Training and evaluation script for the NeuroSense AI Alzheimer's Disease Risk Module.
Conducts strict subject-level holdout validation and persists trained model + SHAP explainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.oasis_ai.alzheimer_model import AlzheimerRiskModel, engineer_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroSense AI Alzheimer's Screening Model")
    parser.add_argument("--data", default="data/interim/oasis.csv", help="Path to oasis dataset CSV")
    parser.add_argument("--output", default="models/neurosense_alzheimer.joblib", help="Model artifact path")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    # Binary clinical risk: 1 = Demented / Converted, 0 = Nondemented
    y = df["Group"].astype(str).str.strip().replace({"Converted": 1, "Demented": 1, "Nondemented": 0}).astype(int)
    
    # Subject-level grouping prevents visits of the same person from leaking into test
    subjects = df["Subject ID"].astype(str).str.strip()

    # Engineer clinical atrophy & demographic features
    X_engineered = engineer_features(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X_engineered, y, groups=subjects))

    X_train, X_test = X_engineered.iloc[train_idx], X_engineered.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Training on {len(train_idx)} visits ({subjects.iloc[train_idx].nunique()} subjects)")
    print(f"Evaluating on {len(test_idx)} visits ({subjects.iloc[test_idx].nunique()} subjects)")

    ad_engine = AlzheimerRiskModel()
    ad_engine.fit(X_train, y_train)

    # Evaluate on held-out subjects
    test_probs = ad_engine.model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    conf_mat = confusion_matrix(y_test, test_preds).tolist()
    clf_rep = classification_report(y_test, test_preds, target_names=["Nondemented", "Impaired/Demented"], output_dict=True)

    metadata = {
        "model_type": "NeuroSense AI - Alzheimer Risk Engine (XGBoost + SHAP)",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": conf_mat,
        "classification_report": clf_rep,
        "train_subjects": int(subjects.iloc[train_idx].nunique()),
        "test_subjects": int(subjects.iloc[test_idx].nunique()),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "engineered_features": ad_engine.feature_names,
        "default_medians": {
            feat: float(X_engineered[feat].median()) for feat in ad_engine.feature_names
        },
        "target_definition": "0 = Nondemented, 1 = Impaired/Demented (including Converted)",
    }

    ad_engine.save(args.output, metadata)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "alzheimer_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 60)
    print("NEUROSENSE AI: ALZHEIMER'S MODEL EVALUATION")
    print("=" * 60)
    print(f"Holdout Accuracy : {acc:.1%}")
    print(f"Holdout Macro F1 : {macro_f1:.3f}")
    print(f"Holdout ROC-AUC  : {roc_auc:.3f}")
    print(f"Saved model to   : {args.output}")
    print(f"Saved metrics to : reports/alzheimer_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
