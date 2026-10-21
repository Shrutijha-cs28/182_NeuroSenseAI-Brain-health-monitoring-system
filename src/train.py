from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from oasis_ai.pipeline import TARGET, build_pipeline, clean_table, read_table, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the OASIS tabular research prototype")
    parser.add_argument("--input", required=True, help="Path to the OASIS demographics .xlsx or .csv")
    parser.add_argument("--output", default="models/oasis_prototype.joblib", help="Model output path")
    args = parser.parse_args()

    raw = read_table(args.input)
    X, y, subjects = clean_table(raw)
    if y.nunique() < 2:
        raise SystemExit("At least two diagnosis groups are needed to train a classifier.")

    # Keep every visit from a person on only one side of the holdout split.
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=42)
    candidates = list(splitter.split(X, y, groups=subjects))
    split = next(
        ((train, test) for train, test in candidates
         if set(y.iloc[train]) == set(y) and set(y.iloc[test]) == set(y)),
        candidates[0],
    )
    train_idx, test_idx = split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = build_pipeline(X)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique())
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "class_labels": labels,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_subjects": int(subjects.iloc[train_idx].nunique()),
        "test_subjects": int(subjects.iloc[test_idx].nunique()),
        "target": TARGET,
        "excluded_columns": ["Subject ID", "MRI ID", "CDR"],
        "note": "Exploratory subject-level holdout estimate; small, single-source cohort.",
    }
    # Persist classes seen per category field so the demo form can offer valid choices.
    categories = {}
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            categories[col] = sorted(X[col].dropna().astype(str).unique().tolist())
    metadata = {
        **metrics,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": X.select_dtypes(include=[np.number]).columns.tolist(),
        "numeric_defaults": X.select_dtypes(include=[np.number]).median().fillna(0).to_dict(),
        "categories": categories,
        "class_distribution": y.value_counts().to_dict(),
        "data_rows": int(len(X)),
    }
    save_model(model, X.columns.tolist(), args.output, metadata)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Cleaned data is generated locally and excluded from Git by .gitignore.
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X.assign(**{TARGET: y}).to_csv("data/processed/oasis_cleaned.csv", index=False)
    print(json.dumps({k: metadata[k] for k in ("accuracy", "macro_f1", "train_subjects", "test_subjects", "class_distribution")}, indent=2))
    print(f"Saved model to {args.output}")
    print("Saved cleaned table to data/processed/oasis_cleaned.csv")


if __name__ == "__main__":
    main()
