"""
09_lightgbm.py — LightGBM with best feature set (piste F)

Feature set: patient aggregates (05) + clinical/missing (06) + rater means (08).
Compares LightGBM against the HGBR baseline on the same folds.
LightGBM handles categoricals natively: str/object columns are marked as
'category' dtype and passed directly.
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


# ── Feature builders (shared logic from 05/06/08) ────────────────────────────
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
    """Convert object columns to 'category' dtype for LightGBM native handling."""
    df = df.copy()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")
    return df


# ── Fixed folds ───────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))


# ── Compare HGBR vs LightGBM ─────────────────────────────────────────────────
results = {}

for model_name, build_fn in [
    ("HGBR+TableVec", lambda X_tr, X_val: (
        make_pipeline(TableVectorizer(),
                      HistGradientBoostingRegressor(random_state=0)),
        X_tr, X_val
    )),
    ("LightGBM", lambda X_tr, X_val: (
        LGBMRegressor(random_state=0, n_estimators=1000,
                      learning_rate=0.05, num_leaves=127,
                      min_child_samples=20, verbose=-1),
        as_lgbm_df(X_tr), as_lgbm_df(X_val)
    )),
]:
    print(f"\nRunning {model_name}...")
    t0 = time.time()
    fold_rmse = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
        X_tr_raw  = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_tr  = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        X_tr  = build_X(X_tr_raw,  X_df_source=X_tr_raw)
        X_val = build_X(X_val_raw, X_df_source=X_tr_raw)

        model, X_tr_m, X_val_m = build_fn(X_tr, X_val)
        model.fit(X_tr_m, y_tr)
        preds = model.predict(X_val_m)
        rmse  = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
        fold_rmse.append(rmse)
        print(f"  Fold {fold_idx + 1}  RMSE={rmse:.4f}")

    elapsed   = time.time() - t0
    mean_rmse = float(np.mean(fold_rmse))
    std_rmse  = float(np.std(fold_rmse))
    results[model_name] = {"mean": mean_rmse, "std": std_rmse, "time": elapsed}
    print(f"  {model_name}: RMSE {mean_rmse:.4f} +/- {std_rmse:.4f}  ({elapsed:.0f}s)")

print("\n=== 09 Summary ===")
for k, v in results.items():
    print(f"  {k:<25} RMSE {v['mean']:.4f} +/- {v['std']:.4f}  ({v['time']:.0f}s)")
print(f"  08_rater_bias baseline:   RMSE  4.44 +/- 0.08")

# ── Pick best, push CV report ─────────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["mean"])
print(f"\nBest: {best_name}")

X_full = build_X(X_raw, X_df_source=X_raw)
folds_full = list(gkf.split(X_full, y, groups=groups))

if best_name == "LightGBM":
    final_model = LGBMRegressor(
        random_state=0, n_estimators=1000,
        learning_rate=0.05, num_leaves=127, min_child_samples=20, verbose=-1
    )
    X_full_m = as_lgbm_df(X_full)
    report_cv = skore.evaluate(final_model, X_full_m, y,
                               splitter=folds_full, n_jobs=-1)
else:
    final_model = make_pipeline(
        TableVectorizer(), HistGradientBoostingRegressor(random_state=0)
    )
    X_full_m = X_full
    report_cv = skore.evaluate(final_model, X_full_m, y,
                               splitter=folds_full, n_jobs=-1)

skore.login()
proj = skore.Project("09-lightgbm", mode="hub", workspace="bob")
proj.put("09_lightgbm", report_cv)
print("Pushed: 09_lightgbm")

# ── Submission ────────────────────────────────────────────────────────────────
X_test_full = build_X(X_test_raw, X_df_source=X_raw)
if best_name == "LightGBM":
    X_test_m = as_lgbm_df(X_test_full)
else:
    X_test_m = X_test_full

final_model.fit(X_full_m, y)
test_preds = np.clip(final_model.predict(X_test_m), 0, 132)

sub_path = PROJECT_ROOT / "submissions" / "09_lightgbm.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows, NaN={np.isnan(test_preds).sum()})")
