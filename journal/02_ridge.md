# 02_ridge — Ridge holdout baseline

**Intent:** First real model. Ridge + median imputer on 8 numeric features, 80/20 holdout. Same split as 01_dummy for direct comparison.

**Features:** `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`. No categoricals.

**Result:** RMSE ≈ 9.5 (80/20 holdout) vs dummy ≈ 16.5. Significant improvement. Not grouped-CV — not directly comparable to later experiments.

**Limitation:** Holdout doesn't respect patient grouping → optimistic estimate. Superseded by GroupKFold in 03_day2.
