# 02_catboost

## Question / hypothesis

Does CatBoost's native categorical handling and residual modelling
(`target − off` when `off` is observed) reduce CV RMSE below the
Exp-01 HistGBR baseline of ~7.45, and does a simple blend with HistGBR
add further value?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Exp 01 result: HistGBR mean CV RMSE ≈ 7.45 (5-fold GroupKFold).
  - `off` (r = 0.87 with target) is the strongest predictor; when
    observed, a residual model directly corrects the bias rather than
    learning it end-to-end. CatBoost handles categoricals and NaN
    natively without an explicit encoding pipeline.
  - User request 2026-10-08.
- **Why this matters:** CatBoost's symmetric trees often outperform
  HGBR on tabular data; the residual formulation makes the remaining
  signal (bias correction) easier to learn; a blend exploits
  decorrelated errors between the two model families.

## Method

- **Files touched:** `experiments/02_catboost.py`
- **Change versus Exp 01:**
  - Model A: `CatBoostRegressor` predicting `target` directly.
    Native categoricals (`cohort`, `gene`, `rater_id`, `sexM`);
    NaN left in numeric columns; same engineered features.
    Config: `loss_function="RMSE"`, `iterations=1000`,
    `learning_rate=0.03`, `depth=7`, `l2_leaf_reg=5`,
    `random_seed=42`, `verbose=False`.
  - Model B: CatBoost residual. For rows with `off` observed
    train `CatBoostRegressor` on `residual = target − off`; predict
    `off + residual_hat`. For rows without `off`, fallback to a
    separate CatBoost direct model. Both components trained strictly
    inside each CV fold. Prediction assembled per-row.
  - Model C: OOF blend `w * CatBoost_direct + (1−w) * HistGBR` for
    `w ∈ {0.25, 0.5, 0.75}`. OOF predictions reused from prior fits;
    no additional refits.
- **Cross-validation:** same 5 precomputed `GroupKFold` folds as
  Exp 01 (recomputed from the same `GroupKFold(n_splits=5)` call on
  the same data + groups).
- **Out of scope:** hyperparameter tuning, Skore upload, SHAP.

## Risks / things that could invalidate the result

- CatBoost's default thread count can trigger the same loky/joblib
  dead-lock as before; must set `thread_count=1` (CatBoost uses
  `thread_count`, not `n_jobs`).
- Residual model splits train/val by `off.notna()` inside each fold;
  if a fold's training slice has too few off-observed rows, the
  residual model may be noisy.
- HistGBR OOF predictions are produced inline (no cache from Exp 01
  run on disk), so HistGBR is refit in this script — 5 extra fits.

## Status

- **State:** approved
- **Approved by user on:** 2026-10-08
- **Headline result:** <fill in after run>
- **Implication for next iteration:** <fill in after run>
