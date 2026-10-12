"""
05_patient_features.py — Patient-level aggregate features (piste A)

Hypothesis: the model sees each visit in isolation; adding patient-level
aggregates (mean off, slope, visit count…) lets it learn each patient's
severity level from their other visits.

All aggregates are computed only from X (no target), so there is no leakage.
For CV: computed on the train split of each fold separately, then applied to
the val split. For test: computed on the full training set.

RMSE is compared to 03_day2 baseline (7.50 ± 0.15, same 5 folds).
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


# ── Feature engineering ───────────────────────────────────────────────────────

def _linslope(s: pd.Series) -> float:
    """Linear slope of s values against 0-based integer positions (fast)."""
    n = len(s.dropna())
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y_vals = s.dropna().values
    # least squares: cov(x,y)/var(x)
    x_c = x - x.mean()
    y_c = y_vals - y_vals.mean()
    denom = (x_c ** 2).sum()
    return float(np.dot(x_c, y_c) / denom) if denom > 0 else np.nan


def add_patient_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute patient-level aggregate features using only X_df itself.
    Since patients are disjoint across GroupKFold splits, each patient's
    aggregates are computed from their OWN visits in the relevant split —
    no cross-split information is used. This is leak-free (no target).

    Works for both train folds and test set: just pass the DataFrame for
    that split (train fold, val fold, or test).
    """
    df = X_df.copy()
    df = df.sort_values(["patient_id", "age"])

    # ── Per-patient aggregates ─────────────────────────────────────────────────
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

    # ── Per-patient slopes ────────────────────────────────────────────────────
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

    # ── Per-row features: visit rank, lag/lead off/on ─────────────────────────
    df["pt_visit_rank"] = df.groupby("patient_id").cumcount()
    df["pt_off_prev"]   = df.groupby("patient_id")["off"].shift(1)
    df["pt_off_next"]   = df.groupby("patient_id")["off"].shift(-1)
    df["pt_on_prev"]    = df.groupby("patient_id")["on"].shift(1)
    df["pt_on_next"]    = df.groupby("patient_id")["on"].shift(-1)

    row_feats = df[["patient_id", "age",
                    "pt_visit_rank",
                    "pt_off_prev", "pt_off_next",
                    "pt_on_prev",  "pt_on_next"]].copy()

    # ── Join ───────────────────────────────────────────────────────────────────
    out = X_df.reset_index()
    out = out.merge(agg, on="patient_id", how="left")
    out = out.merge(row_feats, on=["patient_id", "age"], how="left")
    return out.set_index("Index")


def build_X(X_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Add patient features to X_df, return feature columns."""
    tgt = add_patient_features(X_df)
    pt_cols = [c for c in tgt.columns if c.startswith("pt_")]
    return tgt[features + pt_cols]


# ── Pre-compute fixed 5 GroupKFold folds (same as 03_day2) ───────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))

# ── Model ─────────────────────────────────────────────────────────────────────
def make_model():
    return make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )


# ── Manual OOF loop (needed because feature engineering is fold-dependent) ────
print("Running 05_patient_features  (5-fold GroupKFold OOF)…")
t0 = time.time()

fold_rmse = []
oof_pred = np.full(len(y), np.nan)

for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr_raw = X_raw.iloc[train_idx]
    X_val_raw = X_raw.iloc[val_idx]
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Each split computed independently — patients are disjoint across folds
    X_tr  = build_X(X_tr_raw,  BASE_FEATURES)
    X_val = build_X(X_val_raw, BASE_FEATURES)

    model = make_model()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    oof_pred[val_idx] = preds

    rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
    fold_rmse.append(rmse)
    print(f"  Fold {fold_idx + 1}  RMSE={rmse:.4f}")

elapsed = time.time() - t0
mean_rmse = float(np.mean(fold_rmse))
std_rmse  = float(np.std(fold_rmse))
print(f"\n05_patient_features:  RMSE {mean_rmse:.4f} +/- {std_rmse:.4f}  ({elapsed:.0f}s)")
print("  Baseline (03_day2): RMSE  7.50 +/- 0.15")
print(f"  Delta vs baseline:  {7.50 - mean_rmse:+.4f}")

# ── Push CV report to hub ─────────────────────────────────────────────────────
# Build full enriched X for skore.evaluate (features computed on full train)
X_full = build_X(X_raw, BASE_FEATURES)
pt_extra_cols = [c for c in X_full.columns if c.startswith("pt_")]
all_feat_cols = BASE_FEATURES + pt_extra_cols

# We use pre-built folds on X_full indices (same as X_raw folds)
folds_full = list(gkf.split(X_full, y, groups=groups))

final_model = make_model()
report_cv = skore.evaluate(
    final_model,
    X_full[all_feat_cols],
    y,
    splitter=folds_full,
    n_jobs=-1,
)

skore.login()
proj = skore.Project("05-patient-features", mode="hub", workspace="bob")
proj.put("05_patient_features", report_cv)
print("Pushed: 05_patient_features")

# ── Refit on all train → submission ──────────────────────────────────────────
X_test_full = build_X(X_test_raw, BASE_FEATURES)

final_model.fit(X_full[all_feat_cols], y)
test_preds = np.clip(final_model.predict(X_test_full[all_feat_cols]), 0, 132)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path = submissions_dir / "05_patient_features.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)
print(f"Submission: {sub_path}  ({len(X_test_raw)} rows)")
print(f"NaN in submission: {np.isnan(test_preds).sum()}")

# CV report URL is the hub URL for Kaggle Submission Description.
print("Hub URL: https://skore.probabl.ai/bob/05-patient-features/cross-validations/59594")
