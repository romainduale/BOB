"""
06_clinical_features.py — Clinical and missingness features (piste B)

Added on top of 05_patient_features:
  - years_since_diag: disease duration (age - age_at_diagnosis)
  - miss_on, miss_off, miss_ledd: explicit binary missing indicators
  - visit_type: off/on availability (off_only, on_only, both, neither)
  - on_off_diff: on - off when both present
  - on_off_ratio: on / off
  - ledd_per_year: ledd / years_since_diag

RMSE is compared to 05_patient_features (4.50 +/- 0.09, same 5 folds).
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

# ── Auth ──────────────────────────────────────────────────────────────────────
skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

# ── Data ──────────────────────────────────────────────────────────────────────
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


# ── Patient-level features (copied from 05) ───────────────────────────────────

def _linslope(s: pd.Series) -> float:
    n = len(s.dropna())
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y_vals = s.dropna().values
    x_c = x - x.mean()
    y_c = y_vals - y_vals.mean()
    denom = (x_c ** 2).sum()
    return float(np.dot(x_c, y_c) / denom) if denom > 0 else np.nan


def add_patient_features(X_df: pd.DataFrame) -> pd.DataFrame:
    df = X_df.copy()
    df = df.sort_values(["patient_id", "age"])

    agg = (
        df.groupby("patient_id")
        .agg(
            pt_n_visits= ("age",   "count"),
            pt_age_min=  ("age",   "min"),
            pt_age_span= ("age",   lambda x: x.max() - x.min()),
            pt_off_mean= ("off",   "mean"),
            pt_off_std=  ("off",   "std"),
            pt_off_med=  ("off",   "median"),
            pt_on_mean=  ("on",    "mean"),
            pt_on_std=   ("on",    "std"),
            pt_ledd_mean=("ledd",  "mean"),
            pt_ledd_std= ("ledd",  "std"),
            pt_missing_off_frac=("off", lambda x: x.isna().mean()),
        )
        .reset_index()
    )

    slope_off = (
        df.groupby("patient_id")
        .apply(lambda g: _linslope(g.sort_values("age")["off"]))
        .rename("pt_off_slope")
        .reset_index()
    )
    slope_on = (
        df.groupby("patient_id")
        .apply(lambda g: _linslope(g.sort_values("age")["on"]))
        .rename("pt_on_slope")
        .reset_index()
    )
    agg = agg.merge(slope_off, on="patient_id").merge(slope_on, on="patient_id")

    df["pt_visit_rank"] = df.groupby("patient_id").cumcount()
    df["pt_off_prev"]   = df.groupby("patient_id")["off"].shift(1)
    df["pt_off_next"]   = df.groupby("patient_id")["off"].shift(-1)
    df["pt_on_prev"]    = df.groupby("patient_id")["on"].shift(1)
    df["pt_on_next"]    = df.groupby("patient_id")["on"].shift(-1)

    row_feats = df[["patient_id", "age",
                    "pt_visit_rank",
                    "pt_off_prev", "pt_off_next",
                    "pt_on_prev",  "pt_on_next"]].copy()

    out = X_df.reset_index()
    out = out.merge(agg, on="patient_id", how="left")
    out = out.merge(row_feats, on=["patient_id", "age"], how="left")
    return out.set_index("Index")


# ── Clinical features (new in 06) ─────────────────────────────────────────────

def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinical derived features and missing indicators to df in-place."""
    df = df.copy()

    # Disease duration: how long since diagnosis
    df["years_since_diag"] = df["age"] - df["age_at_diagnosis"]

    # Explicit missing indicators — HGBR handles NaN natively but explicit
    # indicators let the model use the pattern directly as a feature
    df["miss_on"]   = df["on"].isna().astype(int)
    df["miss_off"]  = df["off"].isna().astype(int)
    df["miss_ledd"] = df["ledd"].isna().astype(int)

    # Visit type: which assessments were recorded
    both    = (~df["on"].isna()) & (~df["off"].isna())
    off_only = df["on"].isna()  & (~df["off"].isna())
    on_only  = (~df["on"].isna()) & df["off"].isna()
    df["visit_type"] = "neither"
    df.loc[both,     "visit_type"] = "both"
    df.loc[off_only, "visit_type"] = "off_only"
    df.loc[on_only,  "visit_type"] = "on_only"

    # on - off: magnitude of levodopa effect when both measured
    df["on_off_diff"]  = df["on"] - df["off"]
    # on / off ratio (on is expected to be lower → ratio < 1 typically)
    df["on_off_ratio"] = df["on"] / (df["off"].replace(0, np.nan))
    # LEDD per year of disease duration (dose intensity)
    df["ledd_per_year"] = df["ledd"] / df["years_since_diag"].replace(0, np.nan)

    return df


def build_X(X_df: pd.DataFrame, features: list) -> pd.DataFrame:
    tgt = add_patient_features(X_df)
    tgt = add_clinical_features(tgt)
    pt_cols  = [c for c in tgt.columns if c.startswith("pt_")]
    clin_cols = ["years_since_diag", "miss_on", "miss_off", "miss_ledd",
                 "visit_type", "on_off_diff", "on_off_ratio", "ledd_per_year"]
    all_cols = features + pt_cols + clin_cols
    return tgt[all_cols]


# ── Fixed 5 GroupKFold folds ──────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))


def make_model():
    return make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )


# ── OOF loop ──────────────────────────────────────────────────────────────────
print("Running 06_clinical_features  (5-fold GroupKFold OOF)...")
t0 = time.time()

fold_rmse = []
for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr  = build_X(X_raw.iloc[train_idx],  BASE_FEATURES)
    X_val = build_X(X_raw.iloc[val_idx],    BASE_FEATURES)
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
print(f"\n06_clinical_features:   RMSE {mean_rmse:.4f} +/- {std_rmse:.4f}  ({elapsed:.0f}s)")
print(f"  05_patient_features:  RMSE  4.50 +/- 0.09")
print(f"  Delta vs 05:          {4.50 - mean_rmse:+.4f}")

# ── Push CV report to hub ─────────────────────────────────────────────────────
X_full = build_X(X_raw, BASE_FEATURES)
all_feat_cols = list(X_full.columns)
folds_full = list(gkf.split(X_full, y, groups=groups))

final_model = make_model()
report_cv = skore.evaluate(
    final_model,
    X_full,
    y,
    splitter=folds_full,
    n_jobs=-1,
)

skore.login()
proj = skore.Project("06-clinical-features", mode="hub", workspace="bob")
proj.put("06_clinical_features", report_cv)
print("Pushed: 06_clinical_features")

# ── Refit on all train → submission ──────────────────────────────────────────
X_test_full = build_X(X_test_raw, BASE_FEATURES)

final_model.fit(X_full, y)
test_preds = np.clip(final_model.predict(X_test_full), 0, 132)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path = submissions_dir / "06_clinical_features.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows)")
print(f"NaN in submission: {np.isnan(test_preds).sum()}")
