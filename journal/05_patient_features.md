# 05_patient_features — Patient-level aggregate features (piste A)

**Hypothesis:** Between-patient σ (14.9) >> within-patient σ (6.5). Adding patient-level aggregates lets the model learn each patient's severity level from their other visits.

**Features added (all computed from X only, no target used):**
- `pt_n_visits`: number of visits per patient (proxy for disease stage/duration)
- `pt_visit_rank`: rank of this visit by age within patient (visit number 1, 2, 3…)
- `pt_age_min`, `pt_age_span`: age at first visit, total follow-up span
- `pt_off_mean`, `pt_off_std`, `pt_off_median`: patient's typical OFF score (captures severity)
- `pt_on_mean`, `pt_on_std`: patient's typical ON score
- `pt_ledd_mean`, `pt_ledd_std`: patient's typical LEDD
- `pt_off_slope`: linear trend of OFF score over age (disease progression)
- `pt_on_slope`: linear trend of ON score over age
- `pt_missing_off_frac`: fraction of visits where OFF is missing (informative)
- `pt_off_prev`, `pt_off_next`: OFF score at adjacent visits (sorted by age)
- `pt_on_prev`, `pt_on_next`: ON score at adjacent visits

**No-leak guarantee:** All aggregates computed within each split (train/val) separately. For test, computed on all train. Each patient's own visit is included in its aggregate (valid since we aggregate within-patient and patients don't overlap across train/test).

**Model:** Same HGBR + TableVectorizer as 03_day2 baseline.

**Expected gain:** Large — should collapse most inter-patient variance.
