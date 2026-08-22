# Kaggle Playground Series S6E8 — Smartphone Addiction

Binary classification of `addicted_label` from smartphone-usage features.
`sample_submission.csv` expects probabilities, so the metric is ROC AUC —
submit probabilities, not hard labels.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Layout

```
data/                 competition CSVs (tracked in this repo)
  train.csv           691,369 rows x 14 cols
  test.csv            345,685 rows x 13 cols
  sample_submission.csv
models/               model artifacts (gitignored)
01_eda.ipynb          initial data description and EDA
```

## Data notes

- Target `addicted_label` is imbalanced: ~71% positive, ~29% negative.
  Use stratified K-fold.
- **Every** feature has 4–19% missing values in both train and test. This is
  deliberate in the synthetic Playground data, not corruption. Either use a
  model with native NaN support (LightGBM / XGBoost / CatBoost) or impute and
  add `_isna` indicator columns.
- Strongest linear signals: `daily_screen_time_hours` (r=0.61),
  `weekend_screen_time` (0.59), `social_media_hours` (0.53).
  `age` (0.004) and `notifications_per_day` (-0.01) carry almost none alone.
- Three categorical columns, all low cardinality: `gender`, `stress_level`,
  `academic_work_impact`.
- No duplicate rows. Train and test distributions match, so CV is trustworthy.

## Note on data size

The CSVs are tracked directly in git (~68 MB total, `train.csv` is 45 MB).
If history growth becomes a problem, migrate them to Git LFS.
