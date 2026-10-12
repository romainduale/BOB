# 08_rater_bias — Rater-bias features (piste D)

**Hypothesis:** The 50 raters show RMSE spread from 7.05 (R23) to 8.40 (R03). The worst raters score ~1.4 RMSE points above median. A rater "severity bias" feature (rater's average deviation from the patient's own mean) could help.

**Features added on top of 06 feature set:**
- `rater_off_mean`: mean `off` score given by this rater across all their visits (computed within train split)
- `rater_mean_target_deviation`: rater's average target relative to patient's mean (rater bias proxy)

**Challenge:** For GroupKFold validation, raters can appear in both train and val, so rater-level statistics computed on train ARE valid for val rows. This is a true rater feature, not a patient feature.

**Model:** Same HGBR + TableVectorizer.

**Expected gain:** Small — rater_id is already in TableVectorizer as a categorical feature.
