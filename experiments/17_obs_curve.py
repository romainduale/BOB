"""
17_obs_curve.py — Observation model + WLS per-patient curve features

The latent target per patient follows a near-perfect quadratic in age
(residual 0.21 RMSE). off and on are noisy observations with biases from
rater, timing, and ledd. This experiment:
  1. Fits a bias model (r_off = off - y) on the train fold using rater_id,
     time_since_intake_off, and ledd as predictors.
  2. Corrects each observation by its estimated bias.
  3. Fits a WLS quadratic per patient weighted by inverse-variance.
  4. Exposes curve-based features to LightGBM:
     - curve_pred: WLS quadratic prediction at this visit's age
     - curve_pred_loo: same but leaving this visit out (avoids self-prediction)
     - curve_slope, curve_curv: patient-level trend parameters
     - curve_n_obs: number of observations used for the curve
     - curve_residual: (off or on, corrected) minus curve_pred

All bias/variance learning happens inside the CV fold — no target leakage.

Compares against 16_final baseline (4.23 +/- 0.09).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

X_raw      = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_df       = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv",  index_col="Index")
y      = y_df["target"]
groups = X_raw["patient_id"]

BASE_FEATURES = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "rater_id", "on", "off",
]
CLIN_COLS = [
    "years_since_diag", "miss_on", "miss_off", "miss_ledd",
    "visit_type", "on_off_diff", "on_off_ratio", "ledd_per_year",
]
RATER_COLS = ["rater_off_mean", "rater_on_mean", "rater_n_visits"]
CURVE_COLS = [
    "curve_pred", "curve_pred_loo",
    "curve_slope", "curve_curv",
    "curve_n_obs", "curve_residual_off", "curve_residual_on",
    "curve_uncertainty",
]


# ── Standard feature builders (from 16_final) ────────────────────────────────

def _linslope(s):
    n = len(s.dropna())
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y_v = s.dropna().values
    xc, yc = x - x.mean(), y_v - y_v.mean()
    d = (xc ** 2).sum()
    return float(np.dot(xc, yc) / d) if d > 0 else np.nan


def add_patient_features(X_df):
    df = X_df.copy().sort_values(["patient_id", "age"])
    agg = (
        df.groupby("patient_id").agg(
            pt_n_visits=("age", "count"), pt_age_min=("age", "min"),
            pt_age_span=("age", lambda x: x.max() - x.min()),
            pt_off_mean=("off", "mean"), pt_off_std=("off", "std"),
            pt_off_med=("off", "median"), pt_on_mean=("on", "mean"),
            pt_on_std=("on", "std"), pt_ledd_mean=("ledd", "mean"),
            pt_ledd_std=("ledd", "std"),
            pt_missing_off_frac=("off", lambda x: x.isna().mean()),
        ).reset_index()
    )
    sl_off = df.groupby("patient_id").apply(
        lambda g: _linslope(g.sort_values("age")["off"])
    ).rename("pt_off_slope").reset_index()
    sl_on = df.groupby("patient_id").apply(
        lambda g: _linslope(g.sort_values("age")["on"])
    ).rename("pt_on_slope").reset_index()
    agg = agg.merge(sl_off, on="patient_id").merge(sl_on, on="patient_id")
    df["pt_visit_rank"] = df.groupby("patient_id").cumcount()
    df["pt_off_prev"] = df.groupby("patient_id")["off"].shift(1)
    df["pt_off_next"] = df.groupby("patient_id")["off"].shift(-1)
    df["pt_on_prev"]  = df.groupby("patient_id")["on"].shift(1)
    df["pt_on_next"]  = df.groupby("patient_id")["on"].shift(-1)
    rf = df[["patient_id", "age", "pt_visit_rank",
             "pt_off_prev", "pt_off_next", "pt_on_prev", "pt_on_next"]].copy()
    out = X_df.reset_index().merge(agg, on="patient_id", how="left")
    out = out.merge(rf, on=["patient_id", "age"], how="left")
    return out.set_index("Index")


def add_clinical_features(df):
    df = df.copy()
    df["years_since_diag"] = df["age"] - df["age_at_diagnosis"]
    df["miss_on"]   = df["on"].isna().astype(int)
    df["miss_off"]  = df["off"].isna().astype(int)
    df["miss_ledd"] = df["ledd"].isna().astype(int)
    both     = (~df["on"].isna()) & (~df["off"].isna())
    off_only = df["on"].isna()   & (~df["off"].isna())
    on_only  = (~df["on"].isna()) & df["off"].isna()
    df["visit_type"] = "neither"
    df.loc[both,     "visit_type"] = "both"
    df.loc[off_only, "visit_type"] = "off_only"
    df.loc[on_only,  "visit_type"] = "on_only"
    df["on_off_diff"]   = df["on"] - df["off"]
    df["on_off_ratio"]  = df["on"] / df["off"].replace(0, np.nan)
    df["ledd_per_year"] = df["ledd"] / df["years_since_diag"].replace(0, np.nan)
    return df


def add_rater_features(X_df_target, X_df_source):
    rater_agg = (
        X_df_source.groupby("rater_id").agg(
            rater_off_mean=("off", "mean"),
            rater_on_mean=("on", "mean"),
            rater_n_visits=("age", "count"),
        ).reset_index()
    )
    out = X_df_target.reset_index().merge(rater_agg, on="rater_id", how="left")
    return out.set_index("Index")


def build_base_features(X_df_target, X_df_source=None):
    src = X_df_source if X_df_source is not None else X_df_target
    tgt = add_patient_features(X_df_target)
    tgt = add_clinical_features(tgt)
    tgt = add_rater_features(tgt, src)
    pt_cols = [c for c in tgt.columns if c.startswith("pt_")]
    return tgt[BASE_FEATURES + pt_cols + CLIN_COLS + RATER_COLS]


def as_lgbm_df(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


# ── Observation model + WLS curve features ───────────────────────────────────

class ObsCurveModel:
    """
    Fits observation bias models r_off = off - y and r_on = on - y as a function
    of rater_id (one-hot), time_since_intake, and ledd. Then for each patient
    fits a WLS quadratic y(age) from the corrected observations, with a population
    prior on the coefficients.

    All fitting is on the training split only — no target leakage.
    """

    def __init__(self, poly_alpha: float = 1.0):
        self.poly_alpha = poly_alpha
        self.bias_off_model: Ridge | None = None
        self.bias_on_model: Ridge | None = None
        self.off_resid_std: float = 8.7  # fallback from data analysis
        self.on_resid_std:  float = 9.9
        self.prior_slope:   float = 0.0
        self.prior_curv:    float = 0.0
        self.rater_enc: OneHotEncoder | None = None

    def _make_obs_matrix(self, X_df: pd.DataFrame, source: str) -> np.ndarray:
        """Build feature matrix for the observation bias model."""
        col_t = f"time_since_intake_{source}"  # 'off' or 'on'
        t = X_df[col_t].fillna(X_df[col_t].median() if not X_df[col_t].isna().all() else 4.0).values
        ledd = X_df["ledd"].fillna(X_df["ledd"].median() if not X_df["ledd"].isna().all() else 0.0).values

        # Rater one-hot (seen raters only — unknown raters get all-zero)
        rater_arr = X_df[["rater_id"]].fillna("unknown")
        _transformed = self.rater_enc.transform(rater_arr)
        rater_enc_out = _transformed.toarray() if hasattr(_transformed, "toarray") else _transformed

        feats = np.column_stack([
            t, t ** 2, np.exp(-0.5 * t), ledd, ledd * t,
            rater_enc_out
        ])
        return feats

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Learn bias models from training data. Uses target — call only on train fold."""
        # Fit rater encoder on train
        self.rater_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.rater_enc.fit(X_train[["rater_id"]].fillna("unknown"))

        # Bias for off
        mask_off = X_train["off"].notna()
        if mask_off.sum() >= 10:
            r_off = (X_train.loc[mask_off, "off"] - y_train.loc[mask_off]).values
            F_off = self._make_obs_matrix(X_train.loc[mask_off], "off")
            self.bias_off_model = Ridge(alpha=1.0, fit_intercept=True)
            self.bias_off_model.fit(F_off, r_off)
            resid_off = r_off - self.bias_off_model.predict(F_off)
            self.off_resid_std = float(np.std(resid_off)) + 1e-3

        # Bias for on
        mask_on = X_train["on"].notna()
        if mask_on.sum() >= 10:
            r_on = (X_train.loc[mask_on, "on"] - y_train.loc[mask_on]).values
            F_on = self._make_obs_matrix(X_train.loc[mask_on], "on")
            self.bias_on_model = Ridge(alpha=1.0, fit_intercept=True)
            self.bias_on_model.fit(F_on, r_on)
            resid_on = r_on - self.bias_on_model.predict(F_on)
            self.on_resid_std = float(np.std(resid_on)) + 1e-3

        # Population prior: mean slope and curvature from per-patient fits
        slopes, curvs = [], []
        for pid, grp in X_train.groupby("patient_id"):
            yt = y_train.loc[grp.index].values
            age = grp["age"].values
            if len(age) >= 3:
                c = np.polyfit(age, yt, 2)
                curvs.append(c[0]); slopes.append(c[1])
            elif len(age) >= 2:
                c = np.polyfit(age, yt, 1)
                slopes.append(c[0])
        self.prior_slope = float(np.nanmean(slopes)) if slopes else 2.5
        self.prior_curv  = float(np.nanmean(curvs))  if curvs  else -0.05
        return self

    def _correct_obs(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with off_c, on_c (corrected) and sigma_off, sigma_on."""
        out = X_df.copy()
        # off
        if self.bias_off_model is not None:
            mask = out["off"].notna()
            if mask.any():
                F = self._make_obs_matrix(out.loc[mask], "off")
                bias = self.bias_off_model.predict(F)
                out.loc[mask, "off_c"] = out.loc[mask, "off"] - bias
        if "off_c" not in out.columns:
            out["off_c"] = out["off"]
        out["sigma_off"] = self.off_resid_std

        # on
        if self.bias_on_model is not None:
            mask = out["on"].notna()
            if mask.any():
                F = self._make_obs_matrix(out.loc[mask], "on")
                bias = self.bias_on_model.predict(F)
                out.loc[mask, "on_c"] = out.loc[mask, "on"] - bias
        if "on_c" not in out.columns:
            out["on_c"] = out["on"]
        out["sigma_on"] = self.on_resid_std
        return out

    def _wls_quadratic(self, ages: np.ndarray, obs: np.ndarray,
                       weights: np.ndarray, loo_idx: int | None = None
                       ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Fit ridge-regularized WLS quadratic to (ages, obs) with weights.
        Returns (fitted_values_at_ages, coeff[curv,slope,intercept], posterior_std).
        loo_idx: if given, exclude that observation from fitting.
        """
        n = len(ages)
        if loo_idx is not None:
            keep = [i for i in range(n) if i != loo_idx]
        else:
            keep = list(range(n))

        if len(keep) < 2:
            mean_obs = float(np.nanmean(obs)) if len(obs) > 0 else 30.0
            return np.full(n, mean_obs), np.array([0., 0., mean_obs]), 10.0

        age_c = ages - ages.mean()
        age_sc = age_c / (age_c.std() + 1e-8)
        V = np.column_stack([age_sc ** 2, age_sc, np.ones(n)])

        V_fit = V[keep]
        obs_fit = obs[keep]
        w_fit = weights[keep]

        # WLS with ridge: (V'WV + alpha*diag) coef = V'Wy
        # Prior: push slope and curv toward population values
        prior = np.array([self.prior_curv, self.prior_slope, 0.0])
        alpha = self.poly_alpha
        reg = alpha * np.eye(3)
        reg[2, 2] = 0.0  # no intercept regularization

        Vw = V_fit.T * w_fit
        A = Vw @ V_fit + reg
        b = Vw @ obs_fit + reg @ prior

        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coef = np.array([0., 0., float(np.nanmean(obs))])

        fitted = V @ coef

        # Posterior uncertainty: diagonal of (V'WV + reg)^{-1}
        try:
            cov = np.linalg.inv(A)
            # Variance of prediction at each point: diag(V @ cov @ V')
            # Simplified: use trace of cov as scalar uncertainty
            post_std = float(np.sqrt(np.trace(cov))) * age_c.std()
        except np.linalg.LinAlgError:
            post_std = 5.0

        return fitted, coef, post_std

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Add curve features to X_df. Uses corrected observations."""
        corrected = self._correct_obs(X_df)
        result = X_df.copy()

        curve_pred      = np.full(len(X_df), np.nan)
        curve_pred_loo  = np.full(len(X_df), np.nan)
        curve_slope     = np.full(len(X_df), np.nan)
        curve_curv      = np.full(len(X_df), np.nan)
        curve_n_obs     = np.zeros(len(X_df))
        curve_resid_off = np.full(len(X_df), np.nan)
        curve_resid_on  = np.full(len(X_df), np.nan)
        curve_unc       = np.full(len(X_df), np.nan)

        idx_map = {idx: pos for pos, idx in enumerate(X_df.index)}

        for pid, grp in corrected.groupby("patient_id"):
            grp_s = grp.sort_values("age")
            ages = grp_s["age"].values
            nn   = len(ages)
            positions = [idx_map[i] for i in grp_s.index]

            # Gather corrected observations + weights
            obs_list, w_list = [], []
            for _, row in grp_s.iterrows():
                obs_vals, w_vals = [], []
                if not np.isnan(row.get("off_c", np.nan)):
                    obs_vals.append(row["off_c"])
                    w_vals.append(1.0 / (row["sigma_off"] ** 2 + 1e-6))
                if not np.isnan(row.get("on_c", np.nan)):
                    obs_vals.append(row["on_c"])
                    w_vals.append(1.0 / (row["sigma_on"] ** 2 + 1e-6))
                obs_list.append(np.mean(obs_vals) if obs_vals else np.nan)
                w_list.append(np.mean(w_vals)   if w_vals   else 0.0)

            obs_arr = np.array(obs_list, dtype=float)
            w_arr   = np.array(w_list,   dtype=float)

            # Fill NaN obs with weighted mean
            valid = ~np.isnan(obs_arr)
            n_valid = int(valid.sum())
            if n_valid == 0:
                obs_arr[:] = 30.0  # global mean fallback
                w_arr[:]   = 0.01
            elif n_valid < nn:
                fill = float(np.average(obs_arr[valid], weights=w_arr[valid]))
                obs_arr[~valid] = fill
                w_arr[~valid] = 0.01

            # Full fit
            fitted, coef, post_std = self._wls_quadratic(ages, obs_arr, w_arr)

            for local_i, pos in enumerate(positions):
                curve_pred[pos]   = fitted[local_i]
                # coef = [curv_coef, slope_coef, intercept] in scaled coords
                # We expose the polynomial coefficients as features
                curve_slope[pos]  = coef[1]   # slope in normalised age units
                curve_curv[pos]   = coef[0]   # curvature in normalised age units
                curve_n_obs[pos]  = n_valid
                curve_unc[pos]    = post_std

            # LOO fit for each visit
            for local_i, pos in enumerate(positions):
                if nn >= 3:
                    fitted_loo, _, _ = self._wls_quadratic(
                        ages, obs_arr, w_arr, loo_idx=local_i
                    )
                    curve_pred_loo[pos] = fitted_loo[local_i]
                else:
                    curve_pred_loo[pos] = fitted[local_i]

            # Residuals (corrected obs - curve pred)
            for local_i, (orig_idx, row) in enumerate(grp_s.iterrows()):
                pos = idx_map[orig_idx]
                if not np.isnan(row.get("off_c", np.nan)):
                    curve_resid_off[pos] = row["off_c"] - fitted[local_i]
                if not np.isnan(row.get("on_c", np.nan)):
                    curve_resid_on[pos] = row["on_c"] - fitted[local_i]

        result["curve_pred"]        = curve_pred
        result["curve_pred_loo"]    = curve_pred_loo
        result["curve_slope"]       = curve_slope
        result["curve_curv"]        = curve_curv
        result["curve_n_obs"]       = curve_n_obs
        result["curve_residual_off"] = curve_resid_off
        result["curve_residual_on"]  = curve_resid_on
        result["curve_uncertainty"] = curve_unc
        return result


def build_all_features(X_df_target, X_df_source, y_source, obs_model=None):
    """Build full feature matrix including curve features. obs_model fitted on source."""
    base = build_base_features(X_df_target, X_df_source)
    if obs_model is None:
        # Create and fit obs model
        obs_model = ObsCurveModel(poly_alpha=1.0)
        obs_model.fit(X_df_source, y_source)
    enriched = obs_model.transform(X_df_target.assign(
        **{c: X_df_target[c] for c in ["off", "on", "ledd",
                                         "time_since_intake_off", "time_since_intake_on",
                                         "rater_id", "age", "patient_id"]}
    ))
    # Join curve features onto base
    curve_feats = enriched[CURVE_COLS]
    return pd.concat([base, curve_feats], axis=1), obs_model


def poly_project(age_arr, pred_arr, degree=2, alpha=1.0):
    n = len(age_arr)
    eff_deg = min(degree, n - 1)
    if eff_deg == 0:
        return np.full(n, pred_arr.mean())
    age_c = age_arr - age_arr.mean()
    age_sc = age_c / (age_c.std() + 1e-8)
    V = np.column_stack([age_sc ** d for d in range(eff_deg + 1)])
    reg = alpha * np.eye(V.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(V.T @ V + reg, V.T @ pred_arr)
    return V @ coef


def project_preds(pred_series, X_meta, degree=2, alpha=1.0):
    result = pred_series.copy()
    for pid, grp in X_meta.groupby("patient_id"):
        idx = grp.index
        smoothed = poly_project(grp["age"].values,
                                pred_series.loc[idx].values, degree, alpha)
        result.loc[idx] = smoothed
    return result


def make_lgbm():
    return LGBMRegressor(
        random_state=0, n_estimators=1000, learning_rate=0.05,
        num_leaves=127, min_child_samples=20, verbose=-1,
    )


# ── Fixed GroupKFold-5 folds ──────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))

# ── OOF loop — compare with and without curve features ───────────────────────
print("Running 17_obs_curve  (5-fold GroupKFold OOF)...")
t0 = time.time()

fold_rmse_base  = []   # LightGBM + base features (16_final equivalent)
fold_rmse_curve = []   # LightGBM + base + curve features
fold_rmse_proj  = []   # + deg-2 projection

oof_proj = np.full(len(y), np.nan)

for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr_raw  = X_raw.iloc[train_idx]
    X_val_raw = X_raw.iloc[val_idx]
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # ── Build features ──────────────────────────────────────────────────────
    # Base (16_final equivalent)
    X_tr_base  = as_lgbm_df(build_base_features(X_tr_raw,  X_df_source=X_tr_raw))
    X_val_base = as_lgbm_df(build_base_features(X_val_raw, X_df_source=X_tr_raw))

    # With curve features — obs model fitted on train fold only
    X_tr_curve, obs_m = build_all_features(X_tr_raw,  X_tr_raw, y_tr)
    X_val_curve, _    = build_all_features(X_val_raw, X_tr_raw, y_tr, obs_model=obs_m)
    X_tr_curve  = as_lgbm_df(X_tr_curve)
    X_val_curve = as_lgbm_df(X_val_curve)

    # ── Base model ──────────────────────────────────────────────────────────
    m_base = make_lgbm()
    m_base.fit(X_tr_base, y_tr)
    p_base = m_base.predict(X_val_base)
    fold_rmse_base.append(float(np.sqrt(np.mean((y_val.values - p_base) ** 2))))

    # ── Curve model ─────────────────────────────────────────────────────────
    m_curve = make_lgbm()
    m_curve.fit(X_tr_curve, y_tr)
    p_curve = m_curve.predict(X_val_curve)
    fold_rmse_curve.append(float(np.sqrt(np.mean((y_val.values - p_curve) ** 2))))

    # ── + deg-2 projection ──────────────────────────────────────────────────
    p_proj = project_preds(
        pd.Series(p_curve, index=X_val_raw.index),
        X_val_raw, degree=2, alpha=1.0
    ).values
    oof_proj[val_idx] = p_proj
    fold_rmse_proj.append(float(np.sqrt(np.mean((y_val.values - p_proj) ** 2))))

    print(f"  Fold {fold_idx+1}  base={fold_rmse_base[-1]:.4f}  "
          f"curve={fold_rmse_curve[-1]:.4f}  proj={fold_rmse_proj[-1]:.4f}")

elapsed = time.time() - t0

print(f"\n17_obs_curve  ({elapsed:.0f}s):")
for label, folds in [
    ("Base (16_final equiv)", fold_rmse_base),
    ("+ curve features",      fold_rmse_curve),
    ("+ deg-2 projection",    fold_rmse_proj),
]:
    m, s = np.mean(folds), np.std(folds)
    # Paired gain vs base
    paired = [fold_rmse_base[i] - folds[i] for i in range(5)]
    print(f"  {label:<28} RMSE {m:.4f} +/- {s:.4f}  "
          f"paired gain {np.mean(paired):.4f} +/- {np.std(paired):.4f}")

print(f"  16_final baseline:           RMSE  4.2252 +/- 0.0891")

# ── Save OOF file for team ensemble ──────────────────────────────────────────
oof_dir = PROJECT_ROOT / "results" / "oof"
oof_dir.mkdir(parents=True, exist_ok=True)
oof_df = pd.DataFrame({
    "Index":      X_raw.index,
    "patient_id": X_raw["patient_id"].values,
    "y_true":     y.values,
    "y_pred":     oof_proj,
    "fold":       [
        next(fi for fi, (_, vi) in enumerate(folds_raw) if pos in vi)
        for pos in range(len(y))
    ],
})
oof_df.to_csv(oof_dir / "17_obs_curve.csv", index=False)
print(f"OOF saved: {oof_dir}/17_obs_curve.csv")

# ── Push CV report ────────────────────────────────────────────────────────────
# Use full-train features for skore report
X_full_base = as_lgbm_df(build_base_features(X_raw, X_df_source=X_raw))
folds_full  = list(gkf.split(X_full_base, y, groups=groups))
skore_model = make_lgbm()
report_cv   = skore.evaluate(skore_model, X_full_base, y,
                              splitter=folds_full, n_jobs=-1)

# EstimatorReport (80/20 holdout) → URL /estimators/... for Kaggle description
report_est = skore.evaluate(make_lgbm(), X_full_base, y, splitter=0.2)

skore.login()
proj_hub = skore.Project("17-obs-curve", mode="hub", workspace="bob")
proj_hub.put("17_obs_curve_cv",  report_cv)
proj_hub.put("17_obs_curve_est", report_est)
print("Pushed: 17_obs_curve_cv  (cross-validation URL above)")
print("Pushed: 17_obs_curve_est (EstimatorReport — use /estimators/... URL for Kaggle)")

# ── Refit on all train + submission ──────────────────────────────────────────
best_label = "+ deg-2 projection" if np.mean(fold_rmse_proj) <= np.mean(fold_rmse_base) \
    else "Base (16_final equiv)"

print(f"\nRefitting on all training data...")
t_refit = time.time()
obs_model_full = ObsCurveModel(poly_alpha=1.0)
obs_model_full.fit(X_raw, y)
X_full_curve, _ = build_all_features(X_raw, X_raw, y, obs_model=obs_model_full)
X_full_curve_lgbm = as_lgbm_df(X_full_curve)
final_model = make_lgbm()
final_model.fit(X_full_curve_lgbm, y)

X_test_curve, _ = build_all_features(X_test_raw, X_raw, y, obs_model=obs_model_full)
X_test_curve_lgbm = as_lgbm_df(X_test_curve)
test_preds_raw = final_model.predict(X_test_curve_lgbm)

test_preds = np.clip(
    project_preds(pd.Series(test_preds_raw, index=X_test_raw.index),
                  X_test_raw, degree=2, alpha=1.0).values,
    0, 132
)
print(f"  Refit in {time.time() - t_refit:.0f}s")

assert int(np.isnan(test_preds).sum()) == 0

sub_path = PROJECT_ROOT / "submissions" / "17_obs_curve.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows, NaN=0)")
