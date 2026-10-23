"""
Batch processes raw vertical ground reaction force (VGRF) trials from devalab/parkinsonsfromgait
into a cleaned tabular biomarker dataset with subject-level identifiers.
"""
from pathlib import Path
from src.oasis_ai.gait_processor import GaitProcessor


def main():
    proc = GaitProcessor()
    raw_dir = Path("data/raw/gait_force")
    out_csv = Path("data/processed/gait_features.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw gait force trials from {raw_dir}...")
    df = proc.process_directory(raw_dir)
    df.to_csv(out_csv, index=False)

    n_subjects = df["subject_id"].nunique()
    print(f"Extracted features from {len(df)} trials across {n_subjects} unique participants.")
    print("Class distribution:", df["target"].value_counts().to_dict())
    print(f"Saved processed features to {out_csv}")


if __name__ == "__main__":
    main()
