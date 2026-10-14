# Cardiovascular Disease Detection from ECG and Chest X-ray

MRes Applied AI thesis project, University of Greater Manchester.

Two independently trained deep learning models applied to one patient-paired cohort, with non-AI verification of predictions and a sex-stratified fairness audit.

## Results (held-out test set)

| Model | Input | Conditions | Macro AUC |
|---|---|---|---|
| ECGNet (custom 1D CNN) | 12-lead ECG | 7 | 0.907 |
| DenseNet-121 (frozen backbone) | Chest X-ray | 3 | 0.826 |

- **ECG conditions:** atrial fibrillation, sinus tachycardia, left and right bundle branch block, ST ischaemia, anterior infarct, prolonged QT
- **X-ray conditions:** cardiomegaly, pulmonary oedema, pleural effusion
- **Fairness:** 9 of 10 conditions showed no significant difference between men and women (bootstrap CI); ST ischaemia differed (p = 0.032) and is reported openly.

## Method highlights

- Cohort: 5,218 patients from MIMIC-IV and MIMIC-CXR with both an ECG and a chest X-ray in the same hospital admission
- Patient-level 70/15/15 split to prevent data leakage
- Per-condition threshold calibration
- Non-AI verification: NeuroKit2 waveform delineation (ECG); Otsu segmentation and cardiothoracic ratio (X-ray)

## Repository contents

- `main_ecg.py`, `train_ecg.py` – ECG pipeline and training
- `xray_main.py`, `xray_train.py` – chest X-ray pipeline and training
- `main.py` – main script
- `fairness_auc_chart.png`, `fairness_results_cache.json` – fairness audit results
- `thesis_report.pdf` – full thesis

## Data

No patient data is included in this repository. MIMIC-IV and MIMIC-CXR are available to credentialed users through [PhysioNet](https://physionet.org) under a data use agreement that prohibits redistribution.

## Author

Neelam Fatima, supervised by Dr Anchal Garg.
