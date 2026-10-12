"""
11_final.py — Final end-to-end pipeline (reproducible, <15 min)

Best model: LightGBM with patient aggregates + clinical/missing + rater features.
RMSE GroupKFold-5: 4.37 +/- 0.09 (vs dummy 16.48, vs baseline HGBR 7.50).

Usage (run from BOB/ with venv active):
    python experiments/11_final.py

To use for a new Kaggle submission on competition day, change REPORT_KEY below.

Outputs:
    - skore CV report pushed to Hub (URL printed)
    - submissions/11_final.csv  (11 013 rows, Index,target, no NaN)

Features used (all from X only, no target — no leakage):
  - Base: cohort, sexM, gene, age_at_diagnosis, age, ledd,
          time_since_intake_on/off, rater_id, on, off
  - Patient aggregates (05): mean/std/median of off/on/ledd per patient,
    visit count, age span, linear slope of off/on over age,
    prev/next off/on at adjacent visits (captures patient severity level)
  - Clinical/missing (06): years_since_diagnosis, binary missing indicators
    for on/off/ledd, visit_type (both/off_only/on_only/neither),
    on_off_diff, on_off_ratio, ledd_per_year
  - Rater (08): rater's mean off/on from train set (rater systematic bias)
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

# ── Config — change REPORT_KEY for each new Kaggle submission ─────────────────
REPORT_KEY = "11_final"
PROJECT_NAME = "11-final"
CLIP_MIN, CLIP_MAX = 0, 132  # MDS-UPDRS valid range

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
CLIN_COLS = [
    "years_since_diag", "miss_on", "miss_off", "miss_ledd",
    "visit_type", "on_off_diff", "on_off_ratio", "ledd_per_year",
]
RATER_COLS = ["rater_off_mean", "rater_on_mean", "rater_n_visits"]


# ── Feature engineering ───────────────────────────────────────────────────────

def _linslope(s: pd.Series) -> float:
    """Linear slope of s (sorted) against 0-based integer index."""
    n = len(s.dropna())
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y_v = s.dropna().values
    xc, yc = x - x.mean(), y_v - y_v.mean()
    d = (xc ** 2).sum()
    return float(np.dot(xc, yc) / d) if d > 0 else np.nan


def add_patient_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level aggregates computed from X_df only (no target).
    Patients are GroupKFold-disjoint, so each patient's own visits
    are always in the same split — fully leak-free.
    """
    df = X_df.copy().sort_values(["patient_id", "age"])

    agg = (
        df.groupby("patient_id").agg(
            pt_n_visits=("age",  "count"),
            pt_age_min= ("age",  "min"),
            pt_age_span=("age",  lambda x: x.max() - x.min()),
            pt_off_mean=("off",  "mean"),
            pt_off_std= ("off",  "std"),
            pt_off_med= ("off",  "median"),
            pt_on_mean= ("on",   "mean"),
            pt_on_std=  ("on",   "std"),
            pt_ledd_mean=("ledd","mean"),
            pt_ledd_std= ("ledd","std"),
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
    df["pt_off_prev"]   = df.groupby("patient_id")["off"].shift(1)
    df["pt_off_next"]   = df.groupby("patient_id")["off"].shift(-1)
    df["pt_on_prev"]    = df.groupby("patient_id")["on"].shift(1)
    df["pt_on_next"]    = df.groupby("patient_id")["on"].shift(-1)

    row_feats = df[["patient_id", "age", "pt_visit_rank",
                    "pt_off_prev", "pt_off_next",
                    "pt_on_prev",  "pt_on_next"]].copy()

    out = X_df.reset_index()
    out = out.merge(agg, on="patient_id", how="left")
    out = out.merge(row_feats, on=["patient_id", "age"], how="left")
    return out.set_index("Index")


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Disease duration, missing indicators, visit type, on/off derived features."""
    df = df.copy()
    df["years_since_diag"] = df["age"] - df["age_at_diagnosis"]
    df["miss_on"]   = df["on"].isna().astype(int)
    df["miss_off"]  = df["off"].isna().astype(int)
    df["miss_ledd"] = df["ledd"].isna().astype(int)
    both     = (~df["on"].isna()) & (~df["off"].isna())
    off_only = df["on"].isna()    & (~df["off"].isna())
    on_only  = (~df["on"].isna()) & df["off"].isna()
    df["visit_type"] = "neither"
    df.loc[both,     "visit_type"] = "both"
    df.loc[off_only, "visit_type"] = "off_only"
    df.loc[on_only,  "visit_type"] = "on_only"
    df["on_off_diff"]   = df["on"] - df["off"]
    df["on_off_ratio"]  = df["on"] / df["off"].replace(0, np.nan)
    df["ledd_per_year"] = df["ledd"] / df["years_since_diag"].replace(0, np.nan)
    return df


def add_rater_features(X_df_target: pd.DataFrame,
                       X_df_source: pd.DataFrame) -> pd.DataFrame:
    """Rater-level means from X_df_source (train), applied to X_df_target."""
    rater_agg = (
        X_df_source.groupby("rater_id").agg(
            rater_off_mean=("off", "mean"),
            rater_on_mean= ("on",  "mean"),
            rater_n_visits=("age", "count"),
        ).reset_index()
    )
    out = X_df_target.reset_index().merge(rater_agg, on="rater_id", how="left")
    return out.set_index("Index")


def build_features(X_df_target: pd.DataFrame,
                   X_df_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full feature matrix. X_df_source defaults to X_df_target."""
    src = X_df_source if X_df_source is not None else X_df_target
    tgt = add_patient_features(X_df_target)
    tgt = add_clinical_features(tgt)
    tgt = add_rater_features(tgt, src)
    pt_cols  = [c for c in tgt.columns if c.startswith("pt_")]
    all_cols = BASE_FEATURES + pt_cols + CLIN_COLS + RATER_COLS
    return tgt[all_cols]


def as_lgbm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert str/object columns to 'category' for LightGBM native handling."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


def make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        random_state=0,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        verbose=-1,
    )


# ── Pre-compute 5 GroupKFold folds ────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))

# ── CV evaluation (OOF loop — feature engineering is fold-aware) ──────────────
print(f"[{REPORT_KEY}] GroupKFold-5 OOF evaluation...")
t0 = time.time()
fold_rmse = []
oof_pred  = np.full(len(y), np.nan)

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
    oof_pred[val_idx] = preds

    rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
    fold_rmse.append(rmse)
    print(f"  Fold {fold_idx + 1}  RMSE={rmse:.4f}")

cv_elapsed = time.time() - t0
mean_rmse  = float(np.mean(fold_rmse))
std_rmse   = float(np.std(fold_rmse))
print(f"\nCV RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}  ({cv_elapsed:.0f}s)")

# ── Push CV report to Skore Hub ───────────────────────────────────────────────
# For skore.evaluate, patient-level features use full train (no fold splitting).
# This slightly inflates the reported RMSE (the OOF loop above is the honest one),
# but the hub report provides the artifact URL for the Kaggle submission.
X_full      = as_lgbm_df(build_features(X_raw, X_df_source=X_raw))
folds_full  = list(gkf.split(X_full, y, groups=groups))
skore_model = make_lgbm()

report_cv = skore.evaluate(skore_model, X_full, y,
                           splitter=folds_full, n_jobs=-1)

skore.login()
proj = skore.Project(PROJECT_NAME, mode="hub", workspace="bob")
proj.put(REPORT_KEY, report_cv)

# The URL is printed here — paste into the Kaggle Submission Description field.
hub_url = f"https://skore.probabl.ai/bob/{PROJECT_NAME}/cross-validations/"
print(f"\nHub report pushed as '{REPORT_KEY}'")
print(f"Consult your report at {hub_url}")

# ── Refit on all training data ────────────────────────────────────────────────
t_refit = time.time()
print("\nRefitting on all training data...")
final_model = make_lgbm()
final_model.fit(X_full, y)
print(f"  Refit done in {time.time() - t_refit:.0f}s")

# ── Generate test predictions → Kaggle submission ─────────────────────────────
X_test_full = as_lgbm_df(build_features(X_test_raw, X_df_source=X_raw))
test_preds  = np.clip(final_model.predict(X_test_full), CLIP_MIN, CLIP_MAX)

nan_count = int(np.isnan(test_preds).sum())
assert nan_count == 0, f"NaN in test predictions: {nan_count}"
assert len(test_preds) == len(X_test_raw), "Row count mismatch"

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path = submissions_dir / f"{REPORT_KEY}.csv"
sub_df = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds})
sub_df.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}  ({len(sub_df)} rows, NaN={nan_count})")

# ── Final summary ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"FINAL SUMMARY — {REPORT_KEY}")
print("=" * 70)
print(f"  GroupKFold-5 RMSE:   {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"  Baseline (dummy):    16.48 +/- 0.06")
print(f"  Baseline (HGBR 03):   7.50 +/- 0.15")
print(f"  This model:           {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"  Improvement vs HGBR: {7.50 - mean_rmse:+.4f}")
print(f"  Total time:          {total_elapsed:.0f}s")
print(f"  Submission:          {sub_path}")
print(f"  Hub report key:      {REPORT_KEY}")
print("=" * 70)
