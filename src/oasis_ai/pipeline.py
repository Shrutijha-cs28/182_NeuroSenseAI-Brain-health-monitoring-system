from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Group"
IDENTIFIERS = {"Subject ID", "MRI ID", "SubjectID", "MRIID"}
# CDR is part of the clinical definition of the labels; exclude it to avoid a
# model that simply learns the assessment used to assign the outcome.
LEAKAGE_COLUMNS = {"CDR"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Input must be an .xlsx or .csv file")
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def clean_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalize values and return features, target labels, and subject groups."""
    frame = frame.copy()
    if TARGET not in frame:
        raise ValueError(f"Required outcome column '{TARGET}' was not found")

    target = frame[TARGET].astype("string").str.strip()
    valid = target.notna() & target.ne("")
    frame = frame.loc[valid].reset_index(drop=True)
    target = target.loc[valid].astype(str).reset_index(drop=True)

    subject_col = next((c for c in ("Subject ID", "SubjectID") if c in frame), None)
    groups = (
        frame[subject_col].astype("string").fillna("unknown").astype(str)
        if subject_col
        else pd.Series(np.arange(len(frame)).astype(str), name="subject_group")
    ).reset_index(drop=True)

    features = frame.drop(
        columns=[c for c in frame.columns if c == TARGET or c in IDENTIFIERS or c in LEAKAGE_COLUMNS]
    )
    # Treat blank strings as missing and make numeric-looking columns numeric.
    for col in features.columns:
        if features[col].dtype == "object" or str(features[col].dtype).startswith("string"):
            features[col] = features[col].replace(r"^\s*$", np.nan, regex=True)
            converted = pd.to_numeric(features[col], errors="coerce")
            non_missing = features[col].notna().sum()
            if non_missing and converted.notna().sum() / non_missing >= 0.9:
                features[col] = converted
            else:
                features[col] = features[col].astype("string").str.strip().astype(object)
    features = features.replace([np.inf, -np.inf], np.nan)
    # Drop columns that contain no usable observations.
    features = features.dropna(axis=1, how="all")
    return features, target, groups


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in features.columns if col not in numeric]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )
    return Pipeline(
        [("preprocess", prep), ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))]
    )


def save_model(model: Pipeline, columns: list[str], path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns, "metadata": metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
