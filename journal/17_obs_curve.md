# 17_obs_curve — Observation model + WLS per-patient curve features (exp 13)

**Hypothesis:** off and on are noisy observations of the latent quadratic target curve,
with biases from rater, timing, and ledd. Correcting each observation and fitting a
WLS quadratic per patient gives better curve estimates → better features for LightGBM.

**Method (all learned inside each CV fold — no leakage):**
1. On train fold: fit bias model r_off = off - y = f(rater_id, t_off, ledd) via
   OLS/ridge; same for r_on. Estimate per-observation variance (heteroscedastic).
2. At inference: correct obs by estimated bias, then fit WLS quadratic per patient
   weighted by inverse-variance. Prior on (intercept, slope, curv) = population mean
   from train fold.
3. Expose as features: curve_pred (WLS at this age), curve_pred_loo (leave-one-out),
   curve_slope, curve_curv, curve_residual, curve_uncertainty.

**Expected gain:** Moderate (+0.1 to +0.3 RMSE over baseline with projection).
