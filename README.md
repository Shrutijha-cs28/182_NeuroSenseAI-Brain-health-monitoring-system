# 182_NeuroSenseAI-Brain-health-monitoring-system

This is a third-year NeuroSenseAI project exploring Alzheimer's and Parkinson's. The current implemented prototype is a small, local Streamlit demo that preprocesses the OASIS longitudinal demographics spreadsheet and trains a subject-level, tabular baseline to predict the spreadsheet's `Group` label. It is a learning/research prototype, **not a diagnostic or screening tool**.

## What this prototype does

- Reads the supplied OASIS longitudinal `.xlsx` table (or a CSV copy).
- Removes participant/session identifiers and `CDR` from inputs. CDR is tied closely to the clinical status label, so using it would make a misleadingly easy prediction.
- Converts numeric-looking values, treats blank values as missing, imputes missing numeric and categorical values inside the training pipeline, scales numeric fields, and one-hot encodes categories.
- Holds out whole participants, so visits from the same person do not occur in both train and evaluation sets.
- Trains a class-balanced logistic-regression baseline, saves a local model, and displays an exploratory prediction form.

On the supplied spreadsheet, the current fixed subject-level holdout run produced **61.2% accuracy** and **0.540 macro F1** across 38 held-out participants (112 participants in training). These are preliminary estimates on this small cohort, not evidence of clinical performance. The three labels in this file are `Nondemented`, `Demented`, and `Converted`.

The current input file is OASIS-2 longitudinal demographics. OASIS describes this cohort as 150 participants with 373 imaging sessions. The `.tar.gz` files named `oasis_cross-sectional_disc*.tar.gz` are OASIS-1 cross-sectional MRI volumes, a different collection; this first prototype does not use those MRI images. The browser `.crdownload` files are incomplete downloads and are not inputs. See the [OASIS-2 description](https://sites.wustl.edu/oasisbrains/home/oasis-2/) and its [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895005/).

### Keep OASIS scans local

The OASIS terms require participant data to remain on access-controlled storage for the authorized user. This GitHub repository is public, so do not commit or upload scans, participant-level derived features, or the source spreadsheet. The repository includes a script to extract only spreadsheet-matched `RAW/*.nifti.hdr` and `RAW/*.nifti.img` files to `data/raw/oasis2_mri/`; `.gitignore` excludes this folder. OASIS recommends keeping its separate dataset projects independent. See the [OASIS data-use terms](https://sites.wustl.edu/oasisbrains/home/access/) and [OASIS-2 download information](https://sites.wustl.edu/oasisbrains/home/oasis-2/).

The supplied `OAS2_RAW_PART2.tar.gz` matched 164 spreadsheet sessions and yielded 1,192 NIfTI files (about 20 GB uncompressed). This is only part of OASIS-2; OASIS also lists a separate `OAS2_RAW_PART1.tar.gz` archive. The current model still uses the demographics table only, so the MRI scans are staged locally for a later image-feature workflow and have not yet been used for training.

## Run locally on Windows

Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the completed spreadsheet into `data/raw/` (or use its current Downloads path) and train:

```powershell
python -m src.train --input "C:\Users\shrut\Downloads\oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
```

The script writes a cleaned local CSV under `data/processed/`, model artifacts under `models/`, and evaluation metrics under `reports/`. Those data and model artifacts are ignored by Git by default.

To extract the matching MRI scans from a locally downloaded OASIS-2 archive, run this from the project folder:

```powershell
python -m src.extract_oasis2 --archive "C:\Users\shrut\Downloads\OAS2_RAW_PART2.tar.gz" --table "C:\Users\shrut\Downloads\oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
```

The extraction can take a while because the tool must read through the compressed archive. The extracted NIfTI files and manifest stay local and are ignored by Git. The current model remains the tabular baseline; MRI-based feature extraction and validation are a separate next step.

## Experimental Alzheimer and MRI modules

The repository also contains an experimental XGBoost/SHAP binary classifier (`src/oasis_ai/alzheimer_model.py` and `src/train_alzheimer.py`) and a basic NIfTI loading helper (`src/oasis_ai/mri_processor.py`). The classifier is not connected to the Streamlit app and is not a validated clinical model. It requires a local, authorized demographics CSV with columns including `Subject ID`, `Group`, `Age`, `EDUC`, `SES`, `MMSE`, `eTIV`, `nWBV`, and `ASF`; MRI processing requires scans already extracted locally. Do not commit participant data, generated model files, or derived participant-level data.

To try the experimental classifier after installing requirements, save a CSV with the required columns as `data/interim/oasis.csv` and run:

```powershell
python -m src.train_alzheimer --data data/interim/oasis.csv
```

The trained model and evaluation output are written locally under `models/` and `reports/`.

Start the interactive demo:

```powershell
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser.

## Put the code on GitHub

This workspace is initialized as a local Git repository. Create an **empty** repository on GitHub, then connect and push it from this folder:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git branch -M main
git push -u origin main
```

Replace the URL with your repository URL. Do not upload the OASIS participant data, downloaded archives, model files, or `.crdownload` files. The `.gitignore` keeps local data and model artifacts out of commits. If a GitHub remote already exists, inspect it with `git remote -v` before adding another one.

## Important limits

- The spreadsheet is small and represents a single research cohort. A participant-level holdout score is only a rough internal check; it does not establish clinical validity or performance on new populations.
- The output predicts the dataset's `Group` label, including its `Converted` category. It does not determine whether a person has Alzheimer’s disease.
- Model scores shown in the demo are not calibrated clinical probabilities. Never use this prototype for diagnosis, treatment, or other care decisions.
- MRI image modeling would require a separate, carefully validated imaging workflow, subject/session matching, preprocessing, and patient-level evaluation. No such image model is included here.
- Check and follow the OASIS data-use terms before sharing derived data or any repository containing data.
