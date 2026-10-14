# 13_intake_time_correction

## Question / hypothesis

UPDRS scores measured far from — or very close to — medication intake are
systematically noisier or shifted relative to the true MDS-UPDRS target because
the measurement captures the patient at a non-representative pharmacodynamic
state.  Dividing the raw `on` / `off` readings by a per-quantile intake-time
ratio curve, fitted on the *training fold only*, should reduce this noise before
the longitudinal feature builder sees it, yielding a lower OOF RMSE than Exp08
(reference OOF RMSE: 3.5019).

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - `time_since_intake_on` / `time_since_intake_off` are present in the dataset
    but not yet exploited by any previous experiment.
  - Clinical intuition: the ON state measurement is most informative when taken
    at the "ON" window (≈ 1–3 h after last dose); very early or late measurements
    mix with the OFF window and introduce noise.
  - `learn_curves` quantifies this empirically: the ratio `score / target`
    varies with quantile of `time_since_intake` — if the variation is
    systematic, correcting for it before feature engineering is justified.
- **Why this matters:** If even a fraction of the ~21 % available
  `time_since_intake_on` rows can be normalised to their pharmacodynamic peak,
  the downstream longitudinal features will be cleaner, and the LGB models
  should generalise better.

## Method

- **Files touched:** `experiments/13_intake_time_correction.py`
- **Change versus Exp08:** A single wrapping layer, `Exp13`, is added on top of
  `ShrunkExpertRegressor` (copied verbatim from Exp08).  No architecture change.

  1. **`learn_curves(X, y, nb=10)`** — for each channel (`on`, `off`):
     - compute the per-row ratio `score / target` for rows where both are
       observed and `target > 10` (avoids division noise near zero);
     - bin `time_since_intake` into 10 equal-frequency quantile bins;
     - record the mean ratio per bin plus the global mean ratio;
     - store `(centers, ratios, overall_mean)`.

  2. **`apply_curves(X, curves)`** — for each channel:
     - look up the bin ratio via `np.interp(t, centers, ratios)`;
     - divide the raw score by `ratio / overall_mean` so the global scale
       is preserved (rows with missing `time_since_intake` are left unchanged).

  3. `Exp13.fit(X, y)`:
     - calls `learn_curves(X, y)` → `self.curves_`;
     - calls `super().fit(apply_curves(X, self.curves_), y)`.

  4. `Exp13.predict(X)`:
     - calls `super().predict(apply_curves(X, self.curves_))`.

- **Cross-validation:** `GroupKFold(n_splits=5)`, `groups=patient_id`.
  Same fold split as all previous experiments.  Enabled by `RUN_FULL_CV=1`.
- **Skore report:** `skore.EstimatorReport` on fold 0 (default `REPORT_FOLD=0`)
  pushed to Hub under key `"13_intake_time_correction"`.

## Leakage audit

| Step | Leakage risk | Verdict |
|---|---|---|
| `learn_curves` sees `y` | Only called inside `fit()` on training labels | ✅ Clean |
| `apply_curves` in `predict` | Uses only `self.curves_` (fitted on train) | ✅ Clean |
| `rater_cal` / `apply_cal` | Fitted inside `fit()`, reused in `predict()` | ✅ Clean |
| Kernel features (`k1.0`, `k2.5`) | Within-patient trajectory only | ✅ Clean |
| `smooth_traj` | Uses model outputs, not labels | ✅ Clean |
| `GroupKFold` split | Patient-disjoint by construction | ✅ Clean |

## Risks / things that could invalidate the result

- `learn_curves` bins are sensitive to the distribution of
  `time_since_intake`.  If test rows cluster in a different range than training
  rows, `np.interp` will extrapolate at the boundary (flat) which is safe but
  uninformative.
- Very few rows have both `on` and `time_since_intake_on` observed (~21 %);
  the correction might be too sparse to matter.
- The correction works at the *raw score* level before rater calibration;
  if the rater effect and the intake-time effect are confounded, the rater
  calibration in `ShrunkExpertRegressor` may partially undo the correction.
- Out-of-scope for this experiment: learning different ratio curves per rater,
  or per cohort / gene subgroup.

## Status

- **State:** done
- **Approved by user on:** 2025-01-XX
- **Headline result:**
  - Fold RMSEs: [3.4311, 3.2943, 3.2258, 3.2621, 3.3145]
  - **Global OOF RMSE: 3.3063** (vs Exp08 reference 3.5019)
  - Delta vs Exp05/Exp08: **−0.1956** — the intake-time correction improves every fold
  - Skore report: https://skore.probabl.ai/bob/bobathon-esilv/estimators/79182
  - Submission: `submissions/13_intake_time_correction.csv` (11 013 rows, 0 NaN,
    min/mean/max: 0.000 / 36.666 / 90.780)
- **Implication for next iteration:** The intake-time normalisation is clearly
  beneficial (−0.20 RMSE vs Exp08).  Promising directions: (a) per-rater or
  per-cohort ratio curves; (b) using `time_since_intake` as an explicit LGB
  feature rather than correcting the raw score; (c) a richer pharmacokinetic
  model (e.g. sinusoidal) instead of a piecewise-linear quantile curve.
