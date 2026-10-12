# 09_lightgbm — LightGBM with best feature set (piste F)

**Hypothesis:** LightGBM handles categoricals natively (no TableVectorizer needed), runs faster, and may find better splits than HGBR on the enriched feature set.

**Feature set:** 05 (patient aggregates) + 06 (clinical/missing) + 08 (rater means). No PK transforms (07 showed no gain).

**Model comparison:**
- HGBR + TableVectorizer (baseline for this experiment): from 08 ≈ 4.44 +/- 0.08
- LightGBM with `categorical_feature='auto'` on str columns

**Result:** See runtime.
