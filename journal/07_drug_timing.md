# 07_drug_timing — Drug timing PK transforms (piste C)

**Hypothesis:** Levodopa follows a PK curve: rapid absorption then exponential decay. The target-off residual varies with `time_since_intake_off` in a non-linear way (13.9 point gap at 4-8h, 7.4 at >8h). Non-linear transforms can capture this.

**Features added on top of 06:**
- `t_off_sq`: time_since_intake_off^2 (quadratic)
- `t_on_sq`: time_since_intake_on^2
- `t_off_exp1`, `t_off_exp2`, `t_off_exp3`: exp(-k*t) for k=0.2, 0.5, 1.0 — PK decay shapes
- `t_off_pk1`, `t_off_pk2`: t*exp(-k*t) — rise-then-fall PK shape
- `ledd_x_t_off_exp`: ledd * exp(-0.5*t_off) — dose-weighted PK effect
- `off_x_t_off_exp`: off * exp(-0.5*t_off) — off score weighted by timing

**Model:** Same HGBR + TableVectorizer.

**Expected gain:** Small-moderate — the PK effect is real but HGBR with raw time features may already partially capture it via tree splits.
