<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed - do not invent facts. Keep "Modelling implications" as
candidate suggestions to weigh when designing the model, not final
decisions.
-->

# EDA: Parkinson's MDS-UPDRS Score Prediction

_Generated from `data/eda.py` on 2024-10-07._

## Dataset at a glance

- **Tables:** 1 (training set = X_train joined with y_train; X_test profiled separately)
- **Shape:** 44,590 rows × 13 columns (train); 11,013 rows × 12 columns (test — no target)
- **Target:** `target` — continuous MDS-UPDRS motor score (regression)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_X_test.html](eda_X_test.html)

## Per-column findings

| Column | dtype | null % | n_unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5,576 (train) / 1,395 (test) | Patient identifier; repeats across rows (longitudinal) |
| `cohort` | string | 0 % | 2 | Two cohorts (A/B) |
| `sexM` | int | 0 % | 2 | Binary sex indicator |
| `gene` | string | **32.4 %** | 4 | Genetic variant; high missingness |
| `age_at_diagnosis` | float | **5.2 %** | 563 | Minor missingness |
| `age` | float | 0 % | 1,150 | Age at visit; continuous |
| `ledd` | float | **36.6 %** | 1,320 | Levodopa equivalent daily dose; substantial missingness |
| `time_since_intake_on` | float | **46.4 %** | 64 | Time since ON medication; ~half missing |
| `time_since_intake_off` | float | **78.8 %** | 178 | Time since OFF medication; ~79 % missing — nearly sparse |
| `rater_id` | string | 0 % | 50 | 50 distinct raters; potential rater effect |
| `on` | float | **29.6 %** | 83 | Score measured ON medication |
| `off` | float | **42.4 %** | 101 | Score measured OFF medication |
| `target` | float | 0 % | 867 | The regression target; no missingness |

Key observations:
- **Heavy missingness** in `time_since_intake_off` (79 %), `time_since_intake_on` (46 %), `ledd` (37 %), `off` (42 %), `gene` (32 %). Any model must handle imputation.
- `patient_id` has far fewer unique values than rows (5,576 patients, 44,590 rows), confirming longitudinal / repeated-measures structure.
- `age_at_diagnosis` and `age` are near-collinear (Pearson r = 0.94) — one may be redundant.
- No datetime-typed columns detected.

## Target

- **Type:** continuous regression target (MDS-UPDRS motor score)
- **Range:** 0 – 109.5
- **Mean:** 37.47 · **SD:** 16.50 · **Median:** 37.3
- **IQR:** 25.6 – 49.3 (range ≈ 23.7 points)
- **Distribution:** roughly symmetric around the mean, with a slight right tail (max 109.5 vs mean 37.5); the histogram shows a bell-shaped peak in the 30–45 range and a thin upper tail.
- **No target missingness.**

Not strongly skewed, so a log/power transform may not be required, but could still help reduce tail error.

## Structure

- **No datetime columns** were detected (skrub found 0 columns with a date-like dtype).
- **Longitudinal / repeated-measures structure:** `patient_id` repeats — on average ~8 rows per patient (44,590 rows / 5,576 unique patients). This means train and test rows can belong to the same patient, creating potential leakage if a simple K-Fold splits rows randomly.
- `rater_id` has 50 unique values and appears in both train and test — rater effects could confound the model.
- No explicit time/visit ordering column in the data (beyond `age` as a proxy for visit time).

## Associations

**With target (`target`):**

| Feature | Pearson r | Cramér V | Notes |
|---|---|---|---|
| `off` | **0.871** | 0.406 | Very strong — MDS-UPDRS OFF score is near-linearly related to target |
| `on` | **0.669** | 0.368 | Strong |
| `ledd` | 0.298 | 0.236 | Moderate |
| `age` | 0.310 | 0.114 | Moderate |
| `age_at_diagnosis` | 0.133 | 0.051 | Weak |

**⚠️ Leakage risk — `off`:** Pearson r = 0.871 between `off` and `target` is extremely high. The `off` column is "MDS-UPDRS score measured OFF medication", while the `target` is presumably the same or very closely related score. This association must be verified: if `off` is essentially a different measurement of the same quantity, including it as a feature would constitute target leakage. Similarly, `on` (r = 0.67) warrants scrutiny.

**Strongest feature–feature links:**
- `age_at_diagnosis` ↔ `age`: Pearson r = 0.94 (collinear — near-redundant).
- `on` ↔ `off`: Pearson r = 0.87 (both are medication-state MDS-UPDRS measurements).
- `ledd` ↔ `on`: r = 0.20 (weaker).

## Modelling implications

- **`patient_id` repeats → use `GroupKFold` on `patient_id`** to avoid a patient's visits leaking across train/val folds. A random K-Fold would be optimistic (same patient in both train and val).
- **No datetime column detected** — `TimeSeriesSplit` is not justified by the data alone. `age` could serve as a proxy visit-time ordering within a patient, but this is domain-dependent.
- **Heavy missingness** in `time_since_intake_off` (79 %), `time_since_intake_on` (46 %), `ledd`, `on`, `off` — an imputation step is required. Skrub's `tabular_pipeline` handles this automatically.
- **Target:** roughly symmetric continuous score, mean ≈ 37. MAE or RMSE are natural metrics; no strong case for a target transform, but worth trying if residuals show heteroscedasticity.
- **`off` and `on` columns:** extremely strong correlation with target. Include with caution — investigate whether these constitute leakage before the baseline experiment.
- **`age_at_diagnosis` + `age` collinearity:** skrub / tree-based models handle this naturally; may drop one for linear models.
- **`rater_id`** (50 categories) and **`gene`** (4 categories, 32 % missing) are medium-cardinality categoricals — skrub's default `GapEncoder` / `MinHashEncoder` handles them.

## Open questions

1. **Is `off` a leakage column?** The `target` definition needs clarification — is it the MDS-UPDRS Part III total measured in an unspecified condition, while `on` and `off` are the same scale measured in defined conditions? If `target` can be derived from `off`, it must be excluded.
2. **Visit ordering:** Is there a visit number or timestamp that should be used for `TimeSeriesSplit` instead of `GroupKFold`?
3. **Missing `gene` values (32 %):** Are these truly unknown or are they a different population (e.g., "sporadic" PD patients)? If the latter, a separate category might be more informative than imputation.
4. **`time_since_intake_on` / `time_since_intake_off`** (46 % / 79 % missing): do these missing values indicate the patient was not in that medication state during that visit, or are they simply not recorded? This affects the imputation strategy (median vs. a sentinel "not applicable" value).
