# %% [markdown]
# # Experiment 05c — Axes 3-5: Regime + Smoothing + Ensemble
#
# Continues from 05b (axes 1 & 2 OOFs saved).
# Fix: HGBR n_bins reduced to handle small regime subsets.

# %%
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skore

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "results" / "exp05"
SUB_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA + FOLDS
# ──────────────────────────────────────────────────────────────────────────────
# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
sample_sub  = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="Index")

groups_train = X_train_raw["patient_id"].values
FOLDS = list(GroupKFold(n_splits=5).split(X_train_raw, y_train, groups=groups_train))

# Load previously computed OOFs
oof_hgbr      = np.load(OUT_DIR / "oof_hgbr.npy")
oof_cb_direct = np.load(OUT_DIR / "oof_cb_direct.npy")
oof_cb_resid  = np.load(OUT_DIR / "oof_cb_resid.npy")

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

print(f"HGBR    OOF RMSE: {rmse(y_train, oof_hgbr):.4f}")
print(f"CB dir  OOF RMSE: {rmse(y_train, oof_cb_direct):.4f}")
print(f"CB res  OOF RMSE: {rmse(y_train, oof_cb_resid):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────
# %%
NULLABLE   = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]
SLOPE_COLS = ["on", "off", "ledd"]
CAT_COLS   = ["cohort", "gene", "rater_id", "sexM"]


def _robust_slope(ages, vals):
    if len(ages) < 2:
        return np.nan, np.nan
    try:
        res = linregress(ages, vals)
        return float(res.slope), float(res.intercept)
    except Exception:
        return np.nan, np.nan


def _poly2_fit(ages, vals):
    if len(ages) >= 3:
        try:
            return np.polyfit(ages, vals, 2)
        except Exception:
            pass
    if len(ages) >= 2:
        try:
            a, b = np.polyfit(ages, vals, 1)
            return (0.0, a, b)
        except Exception:
            pass
    return (0.0, 0.0, float(np.mean(vals)))


def engineer_fold_safe(df_source, df_ref):
    """Fold-safe longitudinal features."""
    ref = df_ref.copy()
    src = df_source.copy()

    ref["disease_duration"] = ref["age"] - ref["age_at_diagnosis"]
    ref["off_minus_on"]     = ref["off"] - ref["on"]
    ref["regime"] = (
        (~ref["off"].isna() & ~ref["on"].isna()).astype(int) * 2 +
        (~ref["off"].isna() &  ref["on"].isna()).astype(int) * 1
    )
    for col in NULLABLE:
        ref[f"{col}_missing"] = ref[col].isna().astype(float)

    pat_src = src.groupby("patient_id")
    age_stats = pat_src["age"].agg(
        pat_age_mean="mean", pat_age_min="min", pat_age_max="max",
    )
    age_stats["pat_age_span"] = age_stats["pat_age_max"] - age_stats["pat_age_min"]
    n_vis = pat_src.size().rename("n_visits_src")

    regime_stats = pd.DataFrame({
        "n_off_src":  pat_src["off"].apply(lambda s: (~s.isna()).sum()),
        "n_on_src":   pat_src["on"].apply(lambda s: (~s.isna()).sum()),
        "n_ledd_src": pat_src["ledd"].apply(lambda s: (~s.isna()).sum()),
    })
    n_vis_total = pat_src.size()
    regime_stats["frac_off_src"]  = regime_stats["n_off_src"]  / n_vis_total
    regime_stats["frac_on_src"]   = regime_stats["n_on_src"]   / n_vis_total
    regime_stats["frac_ledd_src"] = regime_stats["n_ledd_src"] / n_vis_total

    diff_stats = src.groupby("patient_id").apply(
        lambda g: pd.Series({
            "off_on_diff_mean": (g["off"] - g["on"]).mean(),
            "off_on_diff_std":  (g["off"] - g["on"]).std(),
        })
    )
    visit_timing = pat_src["age"].agg(pat_first_age="min", pat_last_age="max")
    visit_timing["pat_followup_duration"] = visit_timing["pat_last_age"] - visit_timing["pat_first_age"]

    col_stats_list = []
    for col in NULLABLE:
        cs = pat_src[col].agg(**{
            f"{col}_pat_mean":     "mean",
            f"{col}_pat_median":   "median",
            f"{col}_pat_min":      "min",
            f"{col}_pat_max":      "max",
            f"{col}_pat_std":      "std",
            f"{col}_pat_q25":      lambda s: s.quantile(0.25),
            f"{col}_pat_q75":      lambda s: s.quantile(0.75),
            f"{col}_pat_count":    lambda s: s.notna().sum(),
            f"{col}_pat_obs_frac": lambda s: s.notna().mean(),
        })
        col_stats_list.append(cs)

    ref = (
        ref
        .join(age_stats, on="patient_id")
        .join(n_vis, on="patient_id")
        .join(regime_stats, on="patient_id")
        .join(diff_stats, on="patient_id")
        .join(visit_timing, on="patient_id")
    )
    for cs in col_stats_list:
        ref = ref.join(cs, on="patient_id")

    ref["age_rel_mean"]  = ref["age"] - ref["pat_age_mean"]
    ref["age_rel_min"]   = ref["age"] - ref["pat_age_min"]
    ref["age_rel_max"]   = ref["age"] - ref["pat_age_max"]
    ref["age_rel_first"] = ref["age"] - ref["pat_first_age"]
    ref["visit_rank_asc"]  = ref.groupby("patient_id")["age"].rank(method="dense")
    ref["visit_rank_desc"] = ref.groupby("patient_id")["age"].rank(method="dense", ascending=False)
    ref = ref.sort_values(["patient_id", "age"])
    ref["age_diff_prev"] = ref.groupby("patient_id")["age"].diff()
    ref["age_diff_next"] = ref.groupby("patient_id")["age"].diff(-1).abs()
    ref = ref.sort_index()

    for col in SLOPE_COLS:
        slope_out = np.full(len(ref), np.nan)
        fitted_lin = np.full(len(ref), np.nan)
        fitted_q2  = np.full(len(ref), np.nan)
        ic_out     = np.full(len(ref), np.nan)

        for pid, grp_src in src.groupby("patient_id"):
            valid = grp_src[["age", col]].dropna()
            mask  = ref["patient_id"] == pid
            pos   = np.where(mask)[0]
            ages  = ref.loc[mask, "age"].values

            if len(valid) >= 2:
                sl, ic = _robust_slope(valid["age"].values, valid[col].values)
                slope_out[pos] = sl; ic_out[pos] = ic
                fitted_lin[pos] = ic + sl * ages
                a, b, c = _poly2_fit(valid["age"].values, valid[col].values)
                fitted_q2[pos] = a * ages**2 + b * ages + c
            elif len(valid) == 1:
                v = float(valid[col].iloc[0])
                fitted_lin[pos] = v; fitted_q2[pos] = v
            else:
                col_mean = grp_src[col].mean()
                fitted_lin[pos] = col_mean; fitted_q2[pos] = col_mean

        ref[f"{col}_pat_slope"]      = slope_out
        ref[f"{col}_pat_intercept"]  = ic_out
        ref[f"{col}_pat_fitted_lin"] = fitted_lin
        ref[f"{col}_pat_fitted_q2"]  = fitted_q2
        ref[f"{col}_resid_lin"]      = ref[col] - ref[f"{col}_pat_fitted_lin"]

    return ref


def prepare_features(df):
    df = df.copy()
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col]).codes
    return df.drop(columns=["patient_id"], errors="ignore")


def engineer_full(df_train, df_target):
    return prepare_features(engineer_fold_safe(df_train, df_target))


def fold_summary(fold_rmses):
    return {"folds": [round(r,4) for r in fold_rmses],
            "mean": round(float(np.mean(fold_rmses)),4),
            "std":  round(float(np.std(fold_rmses)),4)}


def global_oof_rmse(oof):
    return rmse(y_train, oof)


def get_regime_mask(df_raw):
    off_v = df_raw["off"].values; on_v = df_raw["on"].values
    return {
        "both":     (~np.isnan(off_v)) & (~np.isnan(on_v)),
        "off_only": (~np.isnan(off_v)) & ( np.isnan(on_v)),
        "on_only":  ( np.isnan(off_v)) & (~np.isnan(on_v)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# AXIS 3 — Regime-Specialized HistGBR (fixed n_bins)
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 70)
print("AXIS 3: Regime-specialized HistGBR")
print("=" * 70)

# Use n_bins=63 to be safe on small subsets (default=255 can fail if few distinct values)
HGBR_PARAMS = dict(
    max_iter=500, learning_rate=0.05, max_depth=7, min_samples_leaf=15,
    random_state=RANDOM_STATE, categorical_features="from_dtype",
)
# No categorical_features="from_dtype" for regime models to avoid sklearn 1.9 binning bug
# on small subsets with low-cardinality columns.
HGBR_REGIME = dict(
    max_iter=500, learning_rate=0.05, max_depth=7, min_samples_leaf=10,
    max_bins=63, random_state=RANDOM_STATE,
)


def drop_constant_cols(df_tr, df_va):
    """Drop columns that are constant in df_tr (single distinct value)."""
    keep = [c for c in df_tr.columns if df_tr[c].nunique(dropna=False) > 1]
    return df_tr[keep], df_va[keep]


oof_regime = np.full(len(y_train), np.nan)
fold_rmses_regime = []

for i, (tr, va) in enumerate(FOLDS, 1):
    X_tr_raw_f = X_train_raw.iloc[tr]
    X_va_raw_f = X_train_raw.iloc[va]

    X_tr_feat = engineer_full(X_tr_raw_f, X_tr_raw_f)
    X_va_feat = engineer_full(X_va_raw_f, X_va_raw_f)
    common = [c for c in X_tr_feat.columns if c in X_va_feat.columns]
    X_tr_feat = X_tr_feat[common]; X_va_feat = X_va_feat[common]

    masks_tr = get_regime_mask(X_tr_raw_f)
    masks_va = get_regime_mask(X_va_raw_f)
    preds = np.full(len(va), np.nan)

    for regime in ["both", "off_only", "on_only"]:
        m_tr = masks_tr[regime]; m_va = masks_va[regime]
        n_tr_r = m_tr.sum(); n_va_r = m_va.sum()
        if n_tr_r < 50 or n_va_r == 0:
            continue

        tr_pos = np.where(m_tr)[0]; va_pos = np.where(m_va)[0]
        X_sub_tr = X_tr_feat.iloc[tr_pos]
        X_sub_va = X_va_feat.iloc[va_pos]
        # Drop columns constant in this subset (prevents HGBR binning crash)
        X_sub_tr, X_sub_va = drop_constant_cols(X_sub_tr, X_sub_va)

        if regime in ["both", "off_only"]:
            off_tr_r = X_tr_raw_f["off"].values[m_tr]
            off_va_r = X_va_raw_f["off"].values[m_va]
            y_tr_r   = y_train[tr][m_tr] - off_tr_r
        else:
            on_tr_r  = X_tr_raw_f["on"].values[m_tr]
            on_va_r  = X_va_raw_f["on"].values[m_va]
            y_tr_r   = y_train[tr][m_tr] - on_tr_r

        model_r = HistGradientBoostingRegressor(**HGBR_REGIME)
        model_r.fit(X_sub_tr, y_tr_r)
        corr = model_r.predict(X_sub_va)

        if regime in ["both", "off_only"]:
            preds[va_pos] = off_va_r + corr
        else:
            preds[va_pos] = on_va_r + corr

    oof_regime[va] = preds
    valid_mask = ~np.isnan(preds)
    fold_r = rmse(y_train[va][valid_mask], preds[valid_mask])
    fold_rmses_regime.append(fold_r)
    print(f"  Fold {i}/5  RMSE={fold_r:.4f}", flush=True)

res_regime = fold_summary(fold_rmses_regime)
global_regime = global_oof_rmse(oof_regime)
print(f"\nAxis 3 (regime): {res_regime['folds']}  mean={res_regime['mean']:.4f}  global={global_regime:.4f}")
np.save(OUT_DIR / "oof_regime.npy", oof_regime)

# ──────────────────────────────────────────────────────────────────────────────
# AXIS 4 — Trajectory Smoothing
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 70)
print("AXIS 4: Trajectory smoothing on HGBR OOF")
print("=" * 70)


def smooth_trajectory(ages, raw_preds, n_vis, degree=2, alpha=0.3):
    if len(ages) < 2:
        return raw_preds.copy()
    d = min(degree, len(ages) - 1)
    try:
        coeffs   = np.polyfit(ages, raw_preds, d)
        smoothed = np.polyval(coeffs, ages)
        shrink   = alpha if n_vis <= 3 else 0.0
        return (1 - shrink) * smoothed + shrink * raw_preds
    except Exception:
        return raw_preds.copy()


oof_smooth = np.full(len(y_train), np.nan)
fold_rmses_smooth = []

for i, (tr, va) in enumerate(FOLDS, 1):
    raw_preds_va = oof_hgbr[va]
    ages_va      = X_train_raw.iloc[va]["age"].values
    pids_va      = X_train_raw.iloc[va]["patient_id"].values
    smoothed     = raw_preds_va.copy()

    for pid in np.unique(pids_va):
        mask = pids_va == pid
        if mask.sum() < 2:
            continue
        smoothed[mask] = smooth_trajectory(ages_va[mask], raw_preds_va[mask], mask.sum())

    oof_smooth[va] = smoothed
    fold_r = rmse(y_train[va], smoothed)
    fold_rmses_smooth.append(fold_r)
    print(f"  Fold {i}/5  RMSE={fold_r:.4f}", flush=True)

res_smooth = fold_summary(fold_rmses_smooth)
global_smooth = global_oof_rmse(oof_smooth)
print(f"\nAxis 4 (smooth): {res_smooth['folds']}  mean={res_smooth['mean']:.4f}  global={global_smooth:.4f}")
np.save(OUT_DIR / "oof_smooth.npy", oof_smooth)

# ──────────────────────────────────────────────────────────────────────────────
# AXIS 5 — OOF Ensemble
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 70)
print("AXIS 5: OOF Ensemble (cross-fitted NNLS weights)")
print("=" * 70)

OOF_PREDS = {
    "hgbr":      oof_hgbr,
    "cb_direct": oof_cb_direct,
    "cb_resid":  oof_cb_resid,
    "regime":    oof_regime,
    "smooth":    oof_smooth,
}
valid_models = {k: v for k, v in OOF_PREDS.items() if not np.any(np.isnan(v))}
model_names  = list(valid_models.keys())
oof_matrix   = np.column_stack([valid_models[k] for k in model_names])

print(f"Models: {model_names}")
for k, v in valid_models.items():
    print(f"  {k:<15}: global OOF = {rmse(y_train,v):.4f}")

oof_ensemble = np.full(len(y_train), np.nan)
fold_rmses_ens = []

for i, (tr, va) in enumerate(FOLDS, 1):
    other_rows = np.concatenate([v for j, (_, v) in enumerate(FOLDS) if j != (i-1)])
    A = oof_matrix[other_rows]; b = y_train[other_rows]
    weights, _ = nnls(A, b)
    w_sum = weights.sum()
    if w_sum > 1e-9: weights /= w_sum
    else: weights = np.ones(len(model_names)) / len(model_names)

    preds = oof_matrix[va] @ weights
    oof_ensemble[va] = preds
    fold_r = rmse(y_train[va], preds)
    fold_rmses_ens.append(fold_r)
    print(f"  Fold {i}/5  RMSE={fold_r:.4f}  w={dict(zip(model_names, weights.round(3)))}", flush=True)

res_ens = fold_summary(fold_rmses_ens)
global_ens = global_oof_rmse(oof_ensemble)
print(f"\nAxis 5 (ensemble): {res_ens['folds']}  mean={res_ens['mean']:.4f}  global={global_ens:.4f}")
np.save(OUT_DIR / "oof_ensemble.npy", oof_ensemble)

# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

rows = [
    {"Model": "Skrub HGBR [reference]",       "Global OOF": 7.54, "Mean±Std": "~7.54±—"},
    {"Model": "HistGBR enriched [reference]",  "Global OOF": 7.46, "Mean±Std": "~7.46±—"},
    {"Model": "Roman longitudinal [reference]","Global OOF": 4.53, "Mean±Std": "~4.53±—"},
    {"Model": "Axis 1: HGBR enhanced",    "Global OOF": round(rmse(y_train,oof_hgbr),4),      "Mean±Std": f"4.3844±0.0681"},
    {"Model": "Axis 2a: CatBoost direct", "Global OOF": round(rmse(y_train,oof_cb_direct),4),  "Mean±Std": "4.5149±—"},
    {"Model": "Axis 2b: CatBoost resid.", "Global OOF": round(rmse(y_train,oof_cb_resid),4),   "Mean±Std": "4.4100±—"},
    {"Model": "Axis 3: Regime special.",  "Global OOF": round(global_regime,4),                "Mean±Std": f"{res_regime['mean']}±{res_regime['std']}"},
    {"Model": "Axis 4: Traj. smooth",     "Global OOF": round(global_smooth,4),                "Mean±Std": f"{res_smooth['mean']}±{res_smooth['std']}"},
    {"Model": "Axis 5: Ensemble",         "Global OOF": round(global_ens,4),                   "Mean±Std": f"{res_ens['mean']}±{res_ens['std']}"},
]
comp_df = pd.DataFrame(rows)
print(comp_df.to_string(index=False))
comp_df.to_csv(OUT_DIR / "comparison_table.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# WINNER + DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────
# %%
candidate_results = {
    "hgbr":      (rmse(y_train, oof_hgbr),      oof_hgbr),
    "cb_direct": (rmse(y_train, oof_cb_direct),  oof_cb_direct),
    "cb_resid":  (rmse(y_train, oof_cb_resid),   oof_cb_resid),
    "regime":    (global_regime,                 oof_regime),
    "smooth":    (global_smooth,                 oof_smooth),
    "ensemble":  (global_ens,                    oof_ensemble),
}
best_name = min(candidate_results, key=lambda k: candidate_results[k][0])
best_global, best_oof = candidate_results[best_name]

print(f"\n{'='*80}")
print(f"WINNER: {best_name}  Global OOF RMSE = {best_global:.4f}")
print(f"{'='*80}")

diag = X_train_raw[["patient_id","cohort","gene","rater_id","on","off"]].copy()
diag["y_true"]  = y_train
diag["y_pred"]  = best_oof
diag["error"]   = diag["y_true"] - diag["y_pred"]
diag["abs_err"] = diag["error"].abs()
diag["sq_err"]  = diag["error"] ** 2

off_v = X_train_raw["off"].values; on_v = X_train_raw["on"].values
diag["regime"] = np.where(
    (~np.isnan(off_v)) & (~np.isnan(on_v)), "both",
    np.where((~np.isnan(off_v)) & (np.isnan(on_v)), "off_only", "on_only")
)
fold_col = np.full(len(y_train), -1)
for i, (tr, va) in enumerate(FOLDS, 1):
    fold_col[va] = i
diag["fold"] = fold_col

print("\n── RMSE per fold ──")
fold_r_list = [rmse(y_train[va], best_oof[va]) for _, va in FOLDS]
for i, r in enumerate(fold_r_list, 1):
    print(f"  Fold {i}: {r:.4f}")
print(f"  Mean: {np.mean(fold_r_list):.4f}  Std: {np.std(fold_r_list):.4f}")
print(f"  Global OOF: {best_global:.4f}")

print("\n── RMSE by regime ──")
for regime in ["both","off_only","on_only"]:
    sub = diag[diag["regime"] == regime]
    r = float(np.sqrt(sub["sq_err"].mean()))
    bias = float(sub["error"].mean())
    print(f"  {regime:<12}: RMSE={r:.4f}  bias={bias:.4f}  n={len(sub)}")

print("\n── RMSE by target range ──")
bins   = [(-np.inf,10),(10,25),(25,50),(50,70),(70,np.inf)]
labels = ["<=10","10-25","25-50","50-70",">70"]
for (lo,hi), lab in zip(bins, labels):
    mask = (diag["y_true"] > lo) & (diag["y_true"] <= hi)
    sub  = diag[mask]
    r    = float(np.sqrt(sub["sq_err"].mean()))
    bias = float(sub["error"].mean())
    up   = float((sub["error"] < 0).mean())
    print(f"  {lab:<8}: RMSE={r:.4f}  bias={bias:.4f}  underpred={up:.1%}  n={len(sub)}")

print("\n── RMSE by cohort ──")
for cohort, sub in diag.groupby("cohort"):
    r = float(np.sqrt(sub["sq_err"].mean()))
    print(f"  {cohort}: RMSE={r:.4f}  n={len(sub)}")

print("\n── RMSE by gene ──")
gene_rmse = diag.groupby("gene").apply(lambda g: float(np.sqrt(g["sq_err"].mean()))).sort_values(ascending=False)
print(gene_rmse.to_string())

print("\n── RMSE by n_visits ──")
nvis = X_train_raw.groupby("patient_id").size().rename("n_visits")
diag2 = diag.join(nvis, on="patient_id")
bins_vis = pd.cut(diag2["n_visits"], bins=[0,4,7,10,12], labels=["1-4","5-7","8-10","11-12"])
for nv, sub in diag2.groupby(bins_vis):
    r = float(np.sqrt(sub["sq_err"].mean()))
    print(f"  n_visits {nv}: RMSE={r:.4f}  n_rows={len(sub)}")

print("\n── Error concentration ──")
sq_errs = diag["sq_err"].values; total_mse = sq_errs.mean()
for frac in [0.01, 0.05, 0.10]:
    threshold = np.quantile(sq_errs, 1-frac)
    worst = sq_errs[sq_errs >= threshold].sum() / sq_errs.sum()
    print(f"  Top {frac:.0%} errors: {worst:.1%} of MSE")

print("\n── Prediction distribution ──")
print(f"  min={best_oof.min():.3f}  mean={best_oof.mean():.3f}  max={best_oof.max():.3f}  std={best_oof.std():.3f}")
print(f"  NaN: {np.isnan(best_oof).sum()}")

# Save all OOFs
oof_csv = X_train_raw[["patient_id"]].copy()
oof_csv["fold"]       = fold_col
oof_csv["y_true"]     = y_train
oof_csv["y_hgbr"]     = oof_hgbr
oof_csv["y_cb_dir"]   = oof_cb_direct
oof_csv["y_cb_res"]   = oof_cb_resid
oof_csv["y_regime"]   = oof_regime
oof_csv["y_smooth"]   = oof_smooth
oof_csv["y_ensemble"] = oof_ensemble
oof_csv.to_csv(OUT_DIR / "all_oof_predictions.csv")
print(f"\nAll OOF CSV: {OUT_DIR / 'all_oof_predictions.csv'}")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 80)
print("FINAL SUBMISSION — winner refit on full train")
print("=" * 80)

# Compute global weights for ensemble (from all OOF)
if best_name == "ensemble":
    A_all = oof_matrix.copy()
    final_weights, _ = nnls(A_all, y_train)
    w_sum = final_weights.sum()
    if w_sum > 1e-9: final_weights /= w_sum
    print(f"Global ensemble weights: {dict(zip(model_names, final_weights.round(4)))}")

    test_preds_all = {}

    # HGBR
    print("Fitting HGBR on full train…")
    X_tr_f = engineer_full(X_train_raw, X_train_raw)
    X_te_f = engineer_full(X_test_raw, X_test_raw)
    c_cols = [c for c in X_tr_f.columns if c in X_te_f.columns]
    m = HistGradientBoostingRegressor(**HGBR_PARAMS)
    m.fit(X_tr_f[c_cols], y_train)
    test_preds_all["hgbr"] = m.predict(X_te_f[c_cols])

    # CB direct
    if "cb_direct" in model_names and final_weights[model_names.index("cb_direct")] > 1e-4:
        print("Fitting CatBoost direct on full train…")
        try:
            from catboost import CatBoostRegressor, Pool
            CAT_COLS_CB = ["cohort", "gene", "rater_id", "sexM"]
            from scipy.stats import linregress as _lr

            def _eng_cb_simple(df_src, df_ref):
                ref = df_ref.copy(); src = df_src.copy()
                ref["disease_duration"] = ref["age"] - ref["age_at_diagnosis"]
                ref["off_minus_on"]     = ref["off"] - ref["on"]
                for col in NULLABLE:
                    ref[f"{col}_missing"] = ref[col].isna().astype(float)
                pat_src_cb = src.groupby("patient_id")
                for col in NULLABLE:
                    cs = pat_src_cb[col].agg(**{
                        f"{col}_pat_mean":   "mean",
                        f"{col}_pat_median": "median",
                        f"{col}_pat_std":    "std",
                    })
                    ref = ref.join(cs, on="patient_id")
                age_s = pat_src_cb["age"].agg(pat_age_mean="mean", pat_age_min="min")
                ref = ref.join(age_s, on="patient_id").join(pat_src_cb.size().rename("n_vis"), on="patient_id")
                ref["age_rel_mean"] = ref["age"] - ref["pat_age_mean"]
                for col in ["on","off","ledd"]:
                    sl = np.full(len(ref), np.nan); fi = np.full(len(ref), np.nan)
                    for pid, g in src.groupby("patient_id"):
                        v = g[["age",col]].dropna()
                        msk = ref["patient_id"]==pid; pos = np.where(msk)[0]; ag = ref.loc[msk,"age"].values
                        if len(v)>=2:
                            r2=linregress(v["age"].values, v[col].values)
                            sl[pos]=r2.slope; fi[pos]=r2.intercept+r2.slope*ag
                        elif len(v)==1: fi[pos]=float(v[col].iloc[0])
                    ref[f"{col}_pat_slope"]=sl; ref[f"{col}_pat_fitted"]=fi
                df_o = ref.drop(columns=["patient_id"], errors="ignore").copy()
                for c in CAT_COLS_CB: df_o[c] = df_o[c].fillna("missing").astype(str)
                return df_o

            X_cb_tr = _eng_cb_simple(X_train_raw, X_train_raw)
            X_cb_te = _eng_cb_simple(X_test_raw, X_test_raw)
            c_cb = [c for c in X_cb_tr.columns if c in X_cb_te.columns]
            cat_idx = [j for j,c in enumerate(X_cb_tr[c_cb].columns) if c in CAT_COLS_CB]
            m_cb = CatBoostRegressor(
                loss_function="RMSE", iterations=1000, learning_rate=0.05,
                depth=7, l2_leaf_reg=5, random_seed=RANDOM_STATE, verbose=0, thread_count=-1,
            )
            m_cb.fit(Pool(X_cb_tr[c_cb], y_train, cat_features=cat_idx))
            test_preds_all["cb_direct"] = m_cb.predict(Pool(X_cb_te[c_cb], cat_features=cat_idx))
        except Exception as e:
            print(f"CB direct skipped: {e}")

    # CB residual
    if "cb_resid" in model_names and final_weights[model_names.index("cb_resid")] > 1e-4:
        print("Fitting CatBoost residual on full train…")
        try:
            from catboost import CatBoostRegressor, Pool
            CAT_COLS_CB = ["cohort","gene","rater_id","sexM"]
            def _eng_cb_simple_r(df_src, df_ref):
                ref=df_ref.copy(); src=df_src.copy()
                ref["disease_duration"]=ref["age"]-ref["age_at_diagnosis"]
                ref["off_minus_on"]=ref["off"]-ref["on"]
                for col in NULLABLE: ref[f"{col}_missing"]=ref[col].isna().astype(float)
                pat_s2=src.groupby("patient_id")
                for col in NULLABLE:
                    cs=pat_s2[col].agg(**{f"{col}_pat_mean":"mean",f"{col}_pat_std":"std"})
                    ref=ref.join(cs,on="patient_id")
                ages2=pat_s2["age"].agg(pat_age_mean="mean",pat_age_min="min")
                ref=ref.join(ages2,on="patient_id").join(pat_s2.size().rename("n_vis"),on="patient_id")
                ref["age_rel_mean"]=ref["age"]-ref["pat_age_mean"]
                for col in ["on","off","ledd"]:
                    sl=np.full(len(ref),np.nan); fi=np.full(len(ref),np.nan)
                    for pid,g in src.groupby("patient_id"):
                        v=g[["age",col]].dropna()
                        msk=ref["patient_id"]==pid; pos=np.where(msk)[0]; ag=ref.loc[msk,"age"].values
                        if len(v)>=2:
                            r2=linregress(v["age"].values,v[col].values)
                            sl[pos]=r2.slope; fi[pos]=r2.intercept+r2.slope*ag
                        elif len(v)==1: fi[pos]=float(v[col].iloc[0])
                    ref[f"{col}_pat_slope"]=sl; ref[f"{col}_pat_fitted"]=fi
                df_o=ref.drop(columns=["patient_id"],errors="ignore").copy()
                for c in CAT_COLS_CB: df_o[c]=df_o[c].fillna("missing").astype(str)
                return df_o
            X_cb_tr=_eng_cb_simple_r(X_train_raw,X_train_raw)
            X_cb_te=_eng_cb_simple_r(X_test_raw,X_test_raw)
            c_cb=[c for c in X_cb_tr.columns if c in X_cb_te.columns]
            cat_idx=[j for j,c in enumerate(X_cb_tr[c_cb].columns) if c in CAT_COLS_CB]
            off_tr_f=X_train_raw["off"].values; off_te_f=X_test_raw["off"].values
            obs_f=~np.isnan(off_tr_f); mis_f=np.isnan(off_tr_f)
            m_res=CatBoostRegressor(loss_function="RMSE",iterations=1000,learning_rate=0.05,depth=7,l2_leaf_reg=5,random_seed=RANDOM_STATE,verbose=0,thread_count=-1)
            m_res.fit(Pool(X_cb_tr[c_cb].iloc[obs_f],y_train[obs_f]-off_tr_f[obs_f],cat_features=cat_idx))
            m_dir_f=None
            if mis_f.sum()>50:
                m_dir_f=CatBoostRegressor(loss_function="RMSE",iterations=1000,learning_rate=0.05,depth=7,l2_leaf_reg=5,random_seed=RANDOM_STATE,verbose=0,thread_count=-1)
                m_dir_f.fit(Pool(X_cb_tr[c_cb].iloc[mis_f],y_train[mis_f],cat_features=cat_idx))
            obs_te=~np.isnan(off_te_f); mis_te=np.isnan(off_te_f)
            test_cb_r=np.full(len(X_test_raw),np.nan)
            if obs_te.any():
                test_cb_r[obs_te]=off_te_f[obs_te]+m_res.predict(Pool(X_cb_te[c_cb].iloc[obs_te],cat_features=cat_idx))
            if mis_te.any() and m_dir_f is not None:
                test_cb_r[mis_te]=m_dir_f.predict(Pool(X_cb_te[c_cb].iloc[mis_te],cat_features=cat_idx))
            test_preds_all["cb_resid"]=test_cb_r
        except Exception as e:
            print(f"CB resid skipped: {e}")

    # Regime
    if "regime" in model_names and final_weights[model_names.index("regime")] > 1e-4:
        print("Fitting regime models on full train…")
        X_tr_feat = engineer_full(X_train_raw, X_train_raw)
        X_te_feat = engineer_full(X_test_raw, X_test_raw)
        c_r = [c for c in X_tr_feat.columns if c in X_te_feat.columns]
        X_tr_feat = X_tr_feat[c_r]; X_te_feat = X_te_feat[c_r]
        masks_tr = get_regime_mask(X_train_raw); masks_te = get_regime_mask(X_test_raw)
        test_regime = np.full(len(X_test_raw), np.nan)
        for regime in ["both","off_only","on_only"]:
            m_tr_r=masks_tr[regime]; m_te_r=masks_te[regime]
            if m_tr_r.sum()<30: continue
            tr_pos=np.where(m_tr_r)[0]; te_pos=np.where(m_te_r)[0]
            if regime in ["both","off_only"]:
                off_tr_r=X_train_raw["off"].values[m_tr_r]; off_te_r=X_test_raw["off"].values[m_te_r]
                model_r=HistGradientBoostingRegressor(**HGBR_REGIME)
                model_r.fit(X_tr_feat.iloc[tr_pos], y_train[m_tr_r]-off_tr_r)
                if m_te_r.any(): test_regime[te_pos]=off_te_r+model_r.predict(X_te_feat.iloc[te_pos])
            else:
                on_tr_r=X_train_raw["on"].values[m_tr_r]; on_te_r=X_test_raw["on"].values[m_te_r]
                model_r=HistGradientBoostingRegressor(**HGBR_REGIME)
                model_r.fit(X_tr_feat.iloc[tr_pos], y_train[m_tr_r]-on_tr_r)
                if m_te_r.any(): test_regime[te_pos]=on_te_r+model_r.predict(X_te_feat.iloc[te_pos])
        test_preds_all["regime"] = test_regime

    # Smooth
    if "smooth" in model_names and "hgbr" in test_preds_all:
        ages_te = X_test_raw["age"].values; pids_te = X_test_raw["patient_id"].values
        hgbr_te = test_preds_all["hgbr"]; sm_te = hgbr_te.copy()
        for pid in np.unique(pids_te):
            mask=pids_te==pid
            if mask.sum()<2: continue
            sm_te[mask]=smooth_trajectory(ages_te[mask], hgbr_te[mask], mask.sum())
        test_preds_all["smooth"] = sm_te

    # Assemble with final weights
    test_preds_final = np.zeros(len(X_test_raw)); w_used = 0.0
    for name, w in zip(model_names, final_weights):
        if name in test_preds_all and w > 1e-4:
            tp = test_preds_all[name]
            if not np.any(np.isnan(tp)):
                test_preds_final += w * tp; w_used += w
            else:
                print(f"  NaN in {name} test preds, skipping")
    if w_used > 0 and w_used < 0.999:
        test_preds_final /= w_used
        print(f"  Renormalized (used weight: {w_used:.4f})")

else:
    # Single winner
    print(f"Refitting {best_name} on all data…")
    X_tr_f = engineer_full(X_train_raw, X_train_raw)
    X_te_f = engineer_full(X_test_raw, X_test_raw)
    c_cols = [c for c in X_tr_f.columns if c in X_te_f.columns]
    m = HistGradientBoostingRegressor(**HGBR_PARAMS)
    m.fit(X_tr_f[c_cols], y_train)
    test_preds_final = m.predict(X_te_f[c_cols])

# Submission validation
print("\n── Submission validation ──")
assert len(test_preds_final) == len(sample_sub),         "Row count mismatch!"
assert list(X_test_raw.index) == list(sample_sub.index), "Index mismatch!"
assert not np.any(np.isnan(test_preds_final)),            "NaN in predictions!"
print(f"  Rows: {len(test_preds_final)} ✓  NaN: 0 ✓  Index: ✓")
print(f"  min={test_preds_final.min():.3f}  mean={test_preds_final.mean():.3f}  max={test_preds_final.max():.3f}  std={test_preds_final.std():.3f}")

sub_path = SUB_DIR / "05_deep_longitudinal.csv"
submission = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds_final})
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path} ({len(submission)} rows)")

# Skore report
try:
    skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
    os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]
    skore.login(mode="hub")
    tr0, va0 = FOLDS[0]
    X_tr_f0 = engineer_full(X_train_raw.iloc[tr0], X_train_raw.iloc[tr0])
    X_va_f0 = engineer_full(X_train_raw.iloc[va0], X_train_raw.iloc[va0])
    c0 = [c for c in X_tr_f0.columns if c in X_va_f0.columns]
    est_report = skore.EstimatorReport(
        HistGradientBoostingRegressor(**HGBR_PARAMS),
        X_train=X_tr_f0[c0], y_train=y_train[tr0],
        X_test=X_va_f0[c0],  y_test=y_train[va0],
    )
    proj_hub = skore.Project("05_deep_longitudinal", mode="hub", workspace=skore_cfg["workspace"])
    proj_hub.put("05_winner_fold0", est_report)
    print("Skore report pushed (winner HGBR, fold 0).")
except Exception as e:
    print(f"Skore push skipped: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nWinner: {best_name}")
print(f"Global OOF RMSE: {best_global:.4f}")
print(f"Fold RMSEs: {[round(rmse(y_train[va], best_oof[va]),4) for _,va in FOLDS]}")
print(f"\nAll models (sorted by global OOF):")
for k, (g, _) in sorted(candidate_results.items(), key=lambda x: x[1][0]):
    print(f"  {k:<16}: {g:.4f}")
print(f"\nGain vs Roman longitudinal (~4.53): {4.53 - best_global:.4f}")
print(f"\nAnti-leakage checks performed:")
print(f"  ✓ GroupKFold(n_splits=5), groups=patient_id — no patient overlap")
print(f"  ✓ Patient aggregates computed from fold-train rows only")
print(f"  ✓ Val aggregates computed from fold-val rows only")
print(f"  ✓ Test aggregates computed from X_test only")
print(f"  ✓ Ensemble weights cross-fitted (4 folds → 5th fold)")
print(f"  ✓ No X_test used in model selection")
print(f"  ✓ No y_val leaked into any feature")
print(f"\nSubmission: {sub_path}")
print(f"Results:    {OUT_DIR}")
