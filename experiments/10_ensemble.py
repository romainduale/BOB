"""
10_ensemble.py — Weighted average ensemble HGBR + LightGBM (piste G)

Builds OOF predictions for both models on the same 5 GroupKFold folds,
then tests: simple average. Only retains ensemble if RMSE improvement
exceeds the fold std (~0.09).

Builds on 09 feature set (patient + clinical + rater features).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

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
CLIN_COLS = ["years_since_diag", "miss_on", "miss_off", "miss_ledd",
             "visit_type", "on_off_diff", "on_off_ratio", "ledd_per_year"]
RATER_COLS = ["rater_off_mean", "rater_on_mean", "rater_n_visits"]


# ── Feature builders ──────────────────────────────────────────────────────────
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


def build_X(X_df_target, X_df_source=None):
    src = X_df_source if X_df_source is not None else X_df_target
    tgt = add_patient_features(X_df_target)
    tgt = add_clinical_features(tgt)
    tgt = add_rater_features(tgt, src)
    pt_cols  = [c for c in tgt.columns if c.startswith("pt_")]
    all_cols = BASE_FEATURES + pt_cols + CLIN_COLS + RATER_COLS
    return tgt[all_cols]


def as_lgbm_df(df):
    df = df.copy()
    for c in df.select_dtypes(["object", "str"]).columns:
        df[c] = df[c].astype("category")
    return df


# ── Fixed folds ───────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))


# ── Collect OOF for each model ────────────────────────────────────────────────
print("Running ensemble OOF (HGBR + LightGBM)...")
t0 = time.time()

oof_hgbr = np.full(len(y), np.nan)
oof_lgbm = np.full(len(y), np.nan)
fold_rmse_hgbr = []
fold_rmse_lgbm = []
fold_rmse_ens  = []

for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr_raw  = X_raw.iloc[train_idx]
    X_val_raw = X_raw.iloc[val_idx]
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    X_tr  = build_X(X_tr_raw,  X_df_source=X_tr_raw)
    X_val = build_X(X_val_raw, X_df_source=X_tr_raw)

    # HGBR
    m_hgbr = make_pipeline(TableVectorizer(),
                            HistGradientBoostingRegressor(random_state=0))
    m_hgbr.fit(X_tr, y_tr)
    p_hgbr = m_hgbr.predict(X_val)
    oof_hgbr[val_idx] = p_hgbr

    # LightGBM
    m_lgbm = LGBMRegressor(random_state=0, n_estimators=1000,
                            learning_rate=0.05, num_leaves=127,
                            min_child_samples=20, verbose=-1)
    m_lgbm.fit(as_lgbm_df(X_tr), y_tr)
    p_lgbm = m_lgbm.predict(as_lgbm_df(X_val))
    oof_lgbm[val_idx] = p_lgbm

    # Ensemble
    p_ens = 0.5 * p_hgbr + 0.5 * p_lgbm

    rmse_h = float(np.sqrt(np.mean((y_val.values - p_hgbr) ** 2)))
    rmse_l = float(np.sqrt(np.mean((y_val.values - p_lgbm) ** 2)))
    rmse_e = float(np.sqrt(np.mean((y_val.values - p_ens)  ** 2)))
    fold_rmse_hgbr.append(rmse_h)
    fold_rmse_lgbm.append(rmse_l)
    fold_rmse_ens.append(rmse_e)
    print(f"  Fold {fold_idx+1}  HGBR={rmse_h:.4f}  LGBM={rmse_l:.4f}  Ens={rmse_e:.4f}")

elapsed = time.time() - t0

print(f"\n10_ensemble results ({elapsed:.0f}s):")
for label, fold_r in [("HGBR",     fold_rmse_hgbr),
                       ("LightGBM", fold_rmse_lgbm),
                       ("Ensemble", fold_rmse_ens)]:
    m, s = np.mean(fold_r), np.std(fold_r)
    print(f"  {label:<12} RMSE {m:.4f} +/- {s:.4f}")

lgbm_mean = np.mean(fold_rmse_lgbm)
ens_mean  = np.mean(fold_rmse_ens)
lgbm_std  = np.std(fold_rmse_lgbm)
improvement = lgbm_mean - ens_mean
print(f"\n  Improvement ensemble over LightGBM: {improvement:.4f} (noise threshold={lgbm_std:.4f})")

# ── Decision: keep ensemble if improvement > noise ────────────────────────────
use_ensemble = improvement > lgbm_std * 0.5  # conservative: half the std
print(f"  Use ensemble: {use_ensemble}")

# ── Push CV report of best ────────────────────────────────────────────────────
X_full      = build_X(X_raw, X_df_source=X_raw)
folds_full  = list(gkf.split(X_full, y, groups=groups))

if use_ensemble:
    # For skore, submit HGBR (the secondary model in the ensemble)
    final_hgbr = make_pipeline(TableVectorizer(),
                                HistGradientBoostingRegressor(random_state=0))
    report_cv = skore.evaluate(final_hgbr, X_full, y,
                               splitter=folds_full, n_jobs=-1)
    skore.login()
    proj = skore.Project("10-ensemble", mode="hub", workspace="bob")
    proj.put("10_ensemble_hgbr", report_cv)

    # LightGBM separately
    final_lgbm = LGBMRegressor(random_state=0, n_estimators=1000,
                                learning_rate=0.05, num_leaves=127,
                                min_child_samples=20, verbose=-1)
    report_lgbm = skore.evaluate(final_lgbm, as_lgbm_df(X_full), y,
                                  splitter=folds_full, n_jobs=-1)
    proj.put("10_ensemble_lgbm", report_lgbm)
    print("Pushed: 10_ensemble_hgbr, 10_ensemble_lgbm")
else:
    final_lgbm = LGBMRegressor(random_state=0, n_estimators=1000,
                                learning_rate=0.05, num_leaves=127,
                                min_child_samples=20, verbose=-1)
    report_cv = skore.evaluate(final_lgbm, as_lgbm_df(X_full), y,
                               splitter=folds_full, n_jobs=-1)
    skore.login()
    proj = skore.Project("10-ensemble", mode="hub", workspace="bob")
    proj.put("10_lgbm_only", report_cv)
    print("Pushed: 10_lgbm_only")

# ── Refit and submission ──────────────────────────────────────────────────────
X_test_full = build_X(X_test_raw, X_df_source=X_raw)

if use_ensemble:
    final_hgbr.fit(X_full, y)
    t_hgbr = final_hgbr.predict(X_test_full)

    final_lgbm.fit(as_lgbm_df(X_full), y)
    t_lgbm = final_lgbm.predict(as_lgbm_df(X_test_full))

    test_preds = np.clip(0.5 * t_hgbr + 0.5 * t_lgbm, 0, 132)
    sub_key = "10_ensemble"
else:
    final_lgbm.fit(as_lgbm_df(X_full), y)
    test_preds = np.clip(final_lgbm.predict(as_lgbm_df(X_test_full)), 0, 132)
    sub_key = "10_lgbm_best"

sub_path = PROJECT_ROOT / "submissions" / f"{sub_key}.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows, NaN={np.isnan(test_preds).sum()})")
