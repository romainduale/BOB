# 06_clinical_features — Clinical and missingness features (piste B)

**Hypothesis:** Missingness pattern carries signal: "off-only" visit (on=NaN) vs "on-only" vs "both" tells us which assessment was done. Also: disease duration (age - age_at_diagnosis) and derived ratios.

**Features added on top of 05's patient features:**
- `years_since_diag`: age − age_at_diagnosis (disease duration proxy)
- `miss_on`, `miss_off`, `miss_ledd`: binary missing indicators (1=NaN)
- `visit_type`: categorical — "both" / "off_only" / "on_only" / "neither"
- `on_off_diff`: on − off when both present (medication effect size)
- `on_off_ratio`: on / off (relative improvement)
- `ledd_per_year`: ledd / years_since_diag (dose intensity over disease duration)

**Model:** Same HGBR + TableVectorizer.

**Expected gain:** Moderate — missingness is informative but the patient aggregates already capture most of the variance.
