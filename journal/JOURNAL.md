# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's MDS-UPDRS score prediction
- **Goal:** Predict "true" OFF MDS-UPDRS motor score per visit (regression, RMSE)
- **Last experiment:** 16_final — done
- **Last result:** LightGBM + patient/clinical/rater features + deg-2 poly projection, GroupKFold-5 RMSE **4.23 ± 0.09**

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2024-10-07
  - env manager: pip+venv - recorded: 2024-10-07
  - CV splitter family: GroupKFold(5) on patient_id — fixed folds pre-computed once
  - skore mode: hub - recorded: 2024-10-07
  - skore hub workspace: bob
  - baseline RMSE (dummy, grouped-CV): 16.48 ± 0.06
  - baseline RMSE (HGBR + TableVectorizer, grouped-CV): 7.50 ± 0.15
  - **best RMSE achieved:** 4.23 ± 0.09 (LightGBM + deg-2 projection, 16_final)

## Data understanding (EDA)

- **Status:** done - 2024-10-07
- **Summary:** 44 590 train rows × 12 cols (+ 11 013 test rows). Continuous MDS-UPDRS regression target (mean ≈ 37, SD ≈ 16.5). `patient_id` repeats (~8 rows/patient) → `GroupKFold` on patient. Heavy missingness: `time_since_intake_off` 79 %, `off` 42 %, `on` 30 %, `ledd` 37 %, `gene` 32 %. `off` corr with target r = 0.87. Between-patient σ ≈ 14.9 >> within-patient σ ≈ 6.5.
- **3 diagnostic hypotheses tested:**
  1. ✅ **Patient severity not captured** → patient-level aggregates: −3.0 RMSE (biggest gain)
  2. ✅ **Missingness is informative** → clinical/missing indicators: −0.03 (marginal but kept)
  3. ✅ **Drug-timing non-linearity** → PK transforms: 0.00 (HGBR/LGBM already capture via tree splits — dropped)

## History

| Stem | Intent (one line) | Status | RMSE CV (mean ± std) | Time | Hub report | Design note |
|---|---|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) — lower-bound baseline | done | 16.48 ± 0.06 | — | [01_dummy](https://skore.probabl.ai/bob/01_dummy/estimators/57306) | — |
| `02_ridge` | Ridge + median imputer, 80/20 holdout | done | ~9.5 (holdout) | — | — | [02_ridge.md](02_ridge.md) |
| `03_day2` | HGBR + TableVectorizer, GroupKFold-5 | done | 7.50 ± 0.15 | — | — | [03_day2.md](03_day2.md) |
| `04_error_analysis` | OOF diagnostics — no model change | done | same as 03 | — | — | [04_error_analysis.md](04_error_analysis.md) |
| `05_patient_features` | Patient-level aggregate features (A) | done | **4.50 ± 0.09** | 32s | [05](https://skore.probabl.ai/bob/05-patient-features/cross-validations/59594) | [05.md](05_patient_features.md) |
| `06_clinical_features` | Clinical + missing indicators (B) | done | 4.47 ± 0.08 | 25s | [06](https://skore.probabl.ai/bob/06-clinical-features/cross-validations/59624) | [06.md](06_clinical_features.md) |
| `07_drug_timing` | PK non-linear transforms (C) | done | 4.47 ± 0.08 | 32s | [07](https://skore.probabl.ai/bob/07-drug-timing/cross-validations/59654) | [07.md](07_drug_timing.md) |
| `08_rater_bias` | Rater mean features (D) | done | 4.44 ± 0.08 | 88s | [08](https://skore.probabl.ai/bob/08-rater-bias/cross-validations/59696) | [08.md](08_rater_bias.md) |
| `09_lightgbm` | LightGBM native categoricals (F) | done | **4.37 ± 0.09** | 120s | [09](https://skore.probabl.ai/bob/09-lightgbm/cross-validations/60292) | [09.md](09_lightgbm.md) |
| `10_ensemble` | HGBR+LightGBM 50/50 avg (G) | done | 4.33 ± 0.08 | 164s | [10](https://skore.probabl.ai/bob/10-ensemble/cross-validations/60300) | [10.md](10_ensemble.md) |
| `11_final` | Final end-to-end pipeline (LightGBM) | done | **4.37 ± 0.09** | 302s | [11](https://skore.probabl.ai/bob/11-final/cross-validations/60493) | — |
| `12_poly_projection` | Deg-2 poly projection of LightGBM preds | done | **4.23 ± 0.09** | 154s | [12](https://skore.probabl.ai/bob/12-poly-projection/cross-validations/61208) | [12.md](12_poly_projection.md) |
| `16_final` | Final pipeline with projection (session 2) | done | **4.23 ± 0.09** | 526s | [16](https://skore.probabl.ai/bob/16-final/cross-validations/61282) | — |

**Decision log:**
- 05 (+3.0): patient aggregates are by far the biggest gain — between-patient variance dominates
- 06 (+0.03): clinical/missing features kept (marginal but consistent direction)
- 07 (0.00): PK drug-timing transforms dropped — tree models already handle non-linearity
- 08 (+0.03): rater means kept (cheap, marginal positive)
- 09 (+0.07): LightGBM beats HGBR on this feature set
- 10 (+0.04): ensemble gain is below noise threshold (< 0.5 × fold std) — dropped
- 11: LightGBM + (05+06+08) features; 4.37
- 12 (+0.145 paired, std 0.017): deg-2 polynomial projection per patient — strong, consistent gain
- 16: 16_final = 11_final + deg-2 projection, reproducible in ~9 min

**Verified data facts (session 2):**
- Per-patient target is near-perfectly quadratic in age: deg-2 residual 0.21 RMSE (deg-1: 1.19, deg-0: 6.06)
- off − y: mean −5.4, std 8.7; rater std across raters 2.6; timing: −13.9 at 4–8h, −7.3 beyond 8h
- on − y: mean −21.8; strong PK dependence: −11.4 at <0.5h, −24.3 at 2–4h
- LEDD correlation with off−y: −0.049 (weak)

## Comparative results table

| Exp | Model | Post-proc | RMSE mean | RMSE std | Paired gain | Time | Keep? |
|-----|-------|-----------|-----------|----------|-------------|------|-------|
| 03 baseline | HGBR+TableVec | none | 7.50 | 0.15 | — | — | baseline |
| 09 | LightGBM | none | 4.37 | 0.09 | — | 120s | yes |
| 12 | LightGBM | deg-1 proj | 4.27 | 0.08 | +0.098±0.013 | 154s | yes |
| **16** | **LightGBM** | **deg-2 proj** | **4.23** | **0.09** | **+0.145±0.017** | 526s | **FINAL** |

## Backlog

| # | Item | Source | Status |
|---|---|---|---|
| 1 | ~~Patient-level aggregate features (A)~~ | hypothesis 1 | done — 05 |
| 2 | ~~Clinical/missing indicators (B)~~ | hypothesis 2 | done — 06 |
| 3 | ~~Drug-timing PK transforms (C)~~ | hypothesis 3 | done — 07, no gain |
| 4 | ~~Rater-bias features (D)~~ | diagnostics: R03/R15/R01 worst | done — 08 |
| 5 | ~~LightGBM comparison (F)~~ | methodology | done — 09 |
| 6 | ~~Poly projection (session 2)~~ | data insight | done — 12/16, +0.145 |
| 7 | Observation model + Bayesian curve features (exp 13) | session 2 plan | open |
| 8 | Optuna hyperparameter tuning (F) | methodology | open |
| 9 | Two-stage patient smoother (E) | methodology | open |
| 10 | ~~Ensemble (G)~~ | methodology | done — 10, no gain vs noise |
