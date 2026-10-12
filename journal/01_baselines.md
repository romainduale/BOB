# 01_baselines

## Question / hypothesis

Can we establish leakage-safe RMSE baselines — from a naive mean predictor to
`HistGradientBoostingRegressor` — using patient-grouped 5-fold CV, and confirm
that the `off` column is worth including despite its 42 % missingness?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - EDA finding: `off` has Pearson r = 0.87 with `target` but is missing for 42 %
    of rows; naive use without grouped CV would leak patient-level signal.
  - User request 2026-10-08: establish four baselines (Dummy, off-proxy, Ridge,
    HistGBR) under identical GroupKFold(5) folds with engineered features.
- **Why this matters:** The baselines fix a RMSE floor (Dummy) and a strong
  ceiling (off-proxy), bound the expected gain from a proper ML model, and
  confirm that grouped CV is necessary to avoid optimistic scores.

## Method

- **Files touched:** `src/parkinsons/`, `experiments/01_baselines.py`
- **Change versus baseline (or previous experiment):** First experiment.
  Four models evaluated under identical `GroupKFold(n_splits=5)` folds with
  `groups=patient_id`. `patient_id` is excluded from all predictive features.
  - **Model 1 — DummyRegressor(strategy="mean")**: global mean of training fold.
  - **Model 2 — off-proxy**: predict `off` when available, else training-fold mean.
  - **Model 3 — Ridge**: median imputation + scaling on numerics;
    missing-indicator + OneHotEncoder(handle_unknown="ignore") on categoricals.
    Fixed `alpha=10`.
  - **Model 4 — HistGradientBoostingRegressor**: OrdinalEncoder for categoricals,
    native NaN handling for numerics; close-to-default hyperparameters.
  - Engineered features: `disease_duration = age − age_at_diagnosis`,
    `off_minus_on = off − on`, missing indicators for `off`, `on`, `ledd`,
    `time_since_intake_on`, `time_since_intake_off`.
- **Cross-validation:** `GroupKFold(n_splits=5)`, `groups=patient_id`.
  Same fold split object passed to all four models.
- **Out of scope for this experiment:** hyperparameter tuning, skrub DataOps
  graph, temporal ordering within patients, rater-effect encoding.

## Risks / things that could invalidate the result

- If `off` itself is post-outcome (i.e., derived from the same debiasing procedure
  that produced `target`), the off-proxy would be leakage. CONTEXT.md describes
  `off` as the *observed, biased* OFF score — distinct from `target` (true OFF) —
  so it is a legitimate, noisy predictor. Verify no exact-match rows dominate.
- GroupKFold does not guarantee equal-sized folds; folds with more patients than
  others may inflate variance.
- `disease_duration` requires non-null `age_at_diagnosis` (5 % missing); computed
  only when both columns are present (else NaN, handled by imputer/native NaN).

## Status

- **State:** approved
- **Approved by user on:** 2026-10-08
- **Headline result:** <fill in after run>
- **Implication for next iteration:** <fill in after run>
