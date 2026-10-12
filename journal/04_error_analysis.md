# 04_error_analysis — OOF diagnostics for 03_day2 best model

**Intent:** Diagnostic only. Identify where the model fails to focus feature engineering effort.

**Method:** Exact same 5-fold GroupKFold folds as 03_day2. OOF predictions collected, errors grouped by patient/cohort/gene/rater.

**Key findings:**
- RMSE cohort A: 7.65, cohort B: 6.62 — cohort B easier (less variance)
- RMSE by gene: GBA+ worst (7.73), LRRK2+ best (7.40)
- Worst raters: R03 (8.40), R15 (8.28), R01 (8.12) — ~1.4 points above median
- Top errors: patient ETNU0107 (GBA+, target ~100–109, pred ~67–70) — severe case systematically under-predicted
- Patient AYWG6818: target=51.1, pred=6.8 → abs_error=44.3 (largest single error)

**Hypotheses generated:**
1. Patient severity not captured → need patient-level aggregates
2. Visit type (off-only/on-only/both) is informative → missingness indicators
3. Drug timing non-linearity → PK curve transforms

**Outputs:** `results/day3_diagnostics/` — fold_stats.csv, oof_predictions.csv, rmse_by_*.csv, top_errors.csv, 4 plots
