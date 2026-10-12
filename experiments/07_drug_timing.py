"""
07_drug_timing.py — Drug timing PK transforms (piste C)

The levodopa effect follows a pharmacokinetic curve. Non-linear transforms
of time_since_intake_off/on capture the residual bias from drug timing.

Builds on 06_clinical_features feature set.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skore
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


# ── Reuse patient features from 05 ───────────────────────────────────────────
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
        df.groupby("patient_id")
        .agg(
            pt_n_visits= ("age",  "count"),
            pt_age_min=  ("age",  "min"),
            pt_age_span= ("age",  lambda x: x.max() - x.min()),
            pt_off_mean= ("off",  "mean"),
            pt_off_std=  ("off",  "std"),
            pt_off_med=  ("off",  "median"),
            pt_on_mean=  ("on",   "mean"),
            pt_on_std=   ("on",   "std"),
            pt_ledd_mean=("ledd", "mean"),
            pt_ledd_std= ("ledd", "std"),
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


# ── Drug timing (PK) features ─────────────────────────────────────────────────

def add_drug_timing_features(df):
    """Non-linear transforms of levodopa timing based on PK decay curve."""
    df = df.copy()
    t_off = df["time_since_intake_off"].values
    t_on  = df["time_since_intake_on"].values

    # Quadratic — captures peak effect region
    df["t_off_sq"] = t_off ** 2
    df["t_on_sq"]  = t_on  ** 2

    # Exponential decay for 3 rate constants — PK off-phase shapes
    for k, label in [(0.2, "slow"), (0.5, "med"), (1.0, "fast")]:
        df[f"t_off_exp_{label}"] = np.exp(-k * np.where(np.isnan(t_off), np.nan, t_off))
        df[f"t_on_exp_{label}"]  = np.exp(-k * np.where(np.isnan(t_on),  np.nan, t_on))

    # Rise-then-fall PK shape: t * exp(-k*t) — peaks at 1/k hours
    for k, label in [(0.5, "med"), (1.0, "fast")]:
        df[f"t_off_pk_{label}"] = (
            np.where(np.isnan(t_off), np.nan, t_off) *
            np.exp(-k * np.where(np.isnan(t_off), np.nan, t_off))
        )

    # Interaction: ledd * PK decay (dose-weighted effect on target)
    ledd = df["ledd"].values
    df["ledd_x_t_off_exp"] = ledd * np.exp(-0.5 * np.where(np.isnan(t_off), np.nan, t_off))

    # off score weighted by timing (off already corr=0.87 with target)
    off = df["off"].values
    df["off_x_t_off_exp"] = off * np.exp(-0.5 * np.where(np.isnan(t_off), np.nan, t_off))

    return df


PK_COLS = [
    "t_off_sq", "t_on_sq",
    "t_off_exp_slow", "t_off_exp_med", "t_off_exp_fast",
    "t_on_exp_slow",  "t_on_exp_med",  "t_on_exp_fast",
    "t_off_pk_med",   "t_off_pk_fast",
    "ledd_x_t_off_exp", "off_x_t_off_exp",
]
CLIN_COLS = ["years_since_diag", "miss_on", "miss_off", "miss_ledd",
             "visit_type", "on_off_diff", "on_off_ratio", "ledd_per_year"]


def build_X(X_df):
    tgt = add_patient_features(X_df)
    tgt = add_clinical_features(tgt)
    tgt = add_drug_timing_features(tgt)
    pt_cols = [c for c in tgt.columns if c.startswith("pt_")]
    all_cols = BASE_FEATURES + pt_cols + CLIN_COLS + PK_COLS
    return tgt[all_cols]


# ── Fixed folds ───────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))


def make_model():
    return make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )


# ── OOF loop ──────────────────────────────────────────────────────────────────
print("Running 07_drug_timing  (5-fold GroupKFold OOF)...")
t0 = time.time()
fold_rmse = []

for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr  = build_X(X_raw.iloc[train_idx])
    X_val = build_X(X_raw.iloc[val_idx])
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = make_model()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse  = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
    fold_rmse.append(rmse)
    print(f"  Fold {fold_idx + 1}  RMSE={rmse:.4f}")

elapsed   = time.time() - t0
mean_rmse = float(np.mean(fold_rmse))
std_rmse  = float(np.std(fold_rmse))
print(f"\n07_drug_timing:         RMSE {mean_rmse:.4f} +/- {std_rmse:.4f}  ({elapsed:.0f}s)")
print(f"  06_clinical:          RMSE  4.47 +/- 0.08")
print(f"  Delta vs 06:          {4.47 - mean_rmse:+.4f}")

# ── Push CV report ────────────────────────────────────────────────────────────
X_full      = build_X(X_raw)
folds_full  = list(gkf.split(X_full, y, groups=groups))
final_model = make_model()
report_cv   = skore.evaluate(final_model, X_full, y, splitter=folds_full, n_jobs=-1)

skore.login()
proj = skore.Project("07-drug-timing", mode="hub", workspace="bob")
proj.put("07_drug_timing", report_cv)
print("Pushed: 07_drug_timing")

# ── Submission ────────────────────────────────────────────────────────────────
X_test_full = build_X(X_test_raw)
final_model.fit(X_full, y)
test_preds = np.clip(final_model.predict(X_test_full), 0, 132)

sub_path = PROJECT_ROOT / "submissions" / "07_drug_timing.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows, NaN={np.isnan(test_preds).sum()})")
