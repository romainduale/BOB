# 12_poly_projection — Polynomial projection of predictions per patient

**Hypothesis:** Each patient's true target follows a near-perfect quadratic in age (residual 0.21 RMSE). Projecting per-visit predictions onto a smooth polynomial per patient removes visit-level noise.

**Method:**
- Run the best LightGBM model (11_final features) with GroupKFold-5 OOF
- For each patient in the val fold, fit a weighted deg-1 or deg-2 polynomial to OOF predictions (age → pred), with L2 regularization scaled by n_visits to handle patients with few visits
- Return the smoothed prediction at each visit's age
- Compare raw OOF vs projected RMSE (paired by fold)

**No-leak guarantee:** Projection uses only OOF predictions (never the target) and the patient's own visit ages. This is a post-processing step applied after the model predicts.

**Expected gain:** Large if the model's visit-to-visit noise is independent (data says projection of 03_day2 OOF: 7.54 → 5.79). With 11_final already at 4.37, the gain should be smaller but still meaningful.
