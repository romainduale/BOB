# 10_ensemble — Weighted average ensemble HGBR + LightGBM (piste G)

**Hypothesis:** HGBR and LightGBM are diverse enough that their predictions are complementary. A simple average should reduce variance.

**Method:** Simple average of HGBR+TableVec and LightGBM OOF predictions. If the ensemble beats LightGBM alone by more than noise (~0.09), keep it.

**Expected gain:** Small (0.0–0.05) since LightGBM dominates. Keep only if delta > noise.
