# 13_curve_features — Bayesian patient curve features (design note — not yet run)

**Intent:** Learn a per-patient quadratic curve y(age) from the corrected observations, then extract:
- `curve_pred`: WLS quadratic prediction at this visit's age (leave-one-out variant)
- `curve_slope`, `curve_curv`: patient-level slope and curvature
- `curve_residual`: obs − curve_pred
- `curve_uncertainty`: posterior variance at this age

**Status:** Planned for next session. Exp 12 (deg-2 poly projection, +0.145) is the current best.
