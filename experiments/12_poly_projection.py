"""
12_poly_projection.py — Polynomial projection of LightGBM predictions per patient

After the LightGBM model predicts each visit, we project each patient's
predictions onto a smooth polynomial in age. This removes visit-level noise
since the true target is nearly quadratic in age (residual 0.21 RMSE).

Post-processing step: uses only (age, oof_pred) for each patient — no target,
no leakage.

Compares: raw LightGBM OOF, deg-1 projection, deg-2 projection (with ridge
regularization for patients with few visits).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]

skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

X_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_df  = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")
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


# ── Feature builders (from 11_final) ─────────────────────────────────────────
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
            pt_n_visits=("age","count"), pt_age_min=("age","min"),
            pt_age_span=("age", lambda x: x.max()-x.min()),
            pt_off_mean=("off","mean"), pt_off_std=("off","std"),
            pt_off_med=("off","median"), pt_on_mean=("on","mean"),
            pt_on_std=("on","std"), pt_ledd_mean=("ledd","mean"),
            pt_ledd_std=("ledd","std"),
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
    rf = df[["patient_id","age","pt_visit_rank",
             "pt_off_prev","pt_off_next","pt_on_prev","pt_on_next"]].copy()
    out = X_df.reset_index().merge(agg, on="patient_id", how="left")
    out = out.merge(rf, on=["patient_id","age"], how="left")
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
            rater_off_mean=("off","mean"),
            rater_on_mean=("on","mean"),
            rater_n_visits=("age","count"),
        ).reset_index()
    )
    out = X_df_target.reset_index().merge(rater_agg, on="rater_id", how="left")
    return out.set_index("Index")


def build_features(X_df_target, X_df_source=None):
    src = X_df_source if X_df_source is not None else X_df_target
    tgt = add_patient_features(X_df_target)
    tgt = add_clinical_features(tgt)
    tgt = add_rater_features(tgt, src)
    pt_cols  = [c for c in tgt.columns if c.startswith("pt_")]
    all_cols = BASE_FEATURES + pt_cols + CLIN_COLS + RATER_COLS
    return tgt[all_cols]


def as_lgbm_df(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


def make_lgbm():
    return LGBMRegressor(
        random_state=0, n_estimators=1000, learning_rate=0.05,
        num_leaves=127, min_child_samples=20, verbose=-1,
    )


# ── Polynomial projection helper ──────────────────────────────────────────────

def poly_project_preds(age_arr: np.ndarray, pred_arr: np.ndarray,
                       degree: int, alpha: float = 1.0) -> np.ndarray:
    """
    Fit a ridge-regularized polynomial of given degree to (age, pred) pairs
    for one patient, return smoothed predictions at each age.
    alpha controls regularization strength (larger = smoother).
    Falls back to lower degree if n < degree+1.
    """
    n = len(age_arr)
    eff_deg = min(degree, n - 1)
    if eff_deg == 0:
        return np.full(n, pred_arr.mean())

    # Center age for numerical stability
    age_c = age_arr - age_arr.mean()
    age_scale = age_c.std() + 1e-8
    age_n = age_c / age_scale

    # Build Vandermonde matrix
    V = np.column_stack([age_n ** d for d in range(eff_deg + 1)])

    # Ridge: (V'V + alpha*I) coef = V'y
    reg = alpha * np.eye(V.shape[1])
    reg[0, 0] = 0.0  # don't regularize intercept
    coef = np.linalg.solve(V.T @ V + reg, V.T @ pred_arr)
    return V @ coef


def project_oof(oof_pred_series: pd.Series, X_meta: pd.DataFrame,
                degree: int, alpha: float = 1.0) -> pd.Series:
    """
    For each patient in X_meta, project OOF predictions onto a polynomial in age.
    Returns smoothed predictions indexed like oof_pred_series.
    """
    result = oof_pred_series.copy()
    for pid, grp in X_meta.groupby("patient_id"):
        idx = grp.index
        age = grp["age"].values
        preds = oof_pred_series.loc[idx].values
        smoothed = poly_project_preds(age, preds, degree=degree, alpha=alpha)
        result.loc[idx] = smoothed
    return result


# ── Fixed folds ───────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))

# ── OOF loop with projection ──────────────────────────────────────────────────
print("Running 12_poly_projection  (5-fold GroupKFold OOF)...")
t0 = time.time()

oof_raw  = np.full(len(y), np.nan)
fold_rmse_raw = []
fold_rmse_d1  = []
fold_rmse_d2  = []

# Store val indices and meta for projection
val_meta_parts = []

for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr_raw  = X_raw.iloc[train_idx]
    X_val_raw = X_raw.iloc[val_idx]
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    X_tr  = as_lgbm_df(build_features(X_tr_raw, X_df_source=X_tr_raw))
    X_val = as_lgbm_df(build_features(X_val_raw, X_df_source=X_tr_raw))

    model = make_lgbm()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    oof_raw[val_idx] = preds

    rmse_raw = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
    fold_rmse_raw.append(rmse_raw)

    # Project deg-1
    pred_s  = pd.Series(preds, index=X_val_raw.index)
    proj_d1 = project_oof(pred_s, X_val_raw, degree=1, alpha=0.1)
    rmse_d1 = float(np.sqrt(np.mean((y_val.values - proj_d1.values) ** 2)))
    fold_rmse_d1.append(rmse_d1)

    # Project deg-2 (with stronger regularization for small n)
    proj_d2 = project_oof(pred_s, X_val_raw, degree=2, alpha=1.0)
    rmse_d2 = float(np.sqrt(np.mean((y_val.values - proj_d2.values) ** 2)))
    fold_rmse_d2.append(rmse_d2)

    print(f"  Fold {fold_idx+1}  raw={rmse_raw:.4f}  deg1={rmse_d1:.4f}  deg2={rmse_d2:.4f}")

elapsed = time.time() - t0

# ── Summary ───────────────────────────────────────────────────────────────────
for label, folds in [("LightGBM raw", fold_rmse_raw),
                      ("+ deg-1 proj", fold_rmse_d1),
                      ("+ deg-2 proj", fold_rmse_d2)]:
    m, s = np.mean(folds), np.std(folds)
    # Paired gain vs raw
    paired = [fold_rmse_raw[i] - folds[i] for i in range(5)]
    print(f"  {label:<18} RMSE {m:.4f} +/- {s:.4f}  "
          f"paired gain: {np.mean(paired):.4f} +/- {np.std(paired):.4f}")

print(f"  Baseline 11_final:   RMSE  4.37 +/- 0.09")
print(f"  Elapsed: {elapsed:.0f}s")

# Decide best
best_mean = min(np.mean(fold_rmse_raw), np.mean(fold_rmse_d1), np.mean(fold_rmse_d2))
if np.mean(fold_rmse_d2) == best_mean:
    best_label = "deg-2"
    best_folds = fold_rmse_d2
elif np.mean(fold_rmse_d1) == best_mean:
    best_label = "deg-1"
    best_folds = fold_rmse_d1
else:
    best_label = "raw"
    best_folds = fold_rmse_raw
print(f"\n  Best: {best_label}  RMSE {best_mean:.4f} +/- {np.std(best_folds):.4f}")

# ── Push CV report ────────────────────────────────────────────────────────────
# Note: skore.evaluate doesn't support post-processing steps natively,
# so we push a CrossValidationReport wrapping the raw LightGBM for the hub.
X_full = as_lgbm_df(build_features(X_raw, X_df_source=X_raw))
folds_full = list(gkf.split(X_full, y, groups=groups))
final_lgbm = make_lgbm()
report_cv = skore.evaluate(final_lgbm, X_full, y, splitter=folds_full, n_jobs=-1)

skore.login()
proj_hub = skore.Project("12-poly-projection", mode="hub", workspace="bob")
proj_hub.put("12_poly_projection", report_cv)
print("Pushed: 12_poly_projection")
print("Hub: https://skore.probabl.ai/bob/12-poly-projection/cross-validations/")

# ── Submission (best projection applied to test preds) ────────────────────────
final_lgbm.fit(X_full, y)
X_test_full = as_lgbm_df(build_features(X_test_raw, X_df_source=X_raw))
test_preds_raw = final_lgbm.predict(X_test_full)

# Apply best projection degree to test predictions
test_pred_s = pd.Series(test_preds_raw, index=X_test_raw.index)
if best_label in ("deg-1", "deg-2"):
    deg = 1 if best_label == "deg-1" else 2
    alpha = 0.1 if best_label == "deg-1" else 1.0
    test_pred_s = project_oof(test_pred_s, X_test_raw, degree=deg, alpha=alpha)

test_preds = np.clip(test_pred_s.values, 0, 132)
assert np.isnan(test_preds).sum() == 0

sub_path = PROJECT_ROOT / "submissions" / "12_poly_projection.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows, NaN={np.isnan(test_preds).sum()})")
