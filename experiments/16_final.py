"""
16_final.py — Final end-to-end pipeline  (session 2, reproducible, ≤15 min)

Best model: LightGBM + patient/clinical/rater features + deg-2 polynomial
projection of predictions per patient.

GroupKFold-5 RMSE:
  - LightGBM raw:       4.37 +/- 0.09  (11_final)
  - + deg-2 projection: 4.23 +/- 0.09  (this script)  ← best

Usage (from BOB/ with venv active):
    python experiments/16_final.py

Change REPORT_KEY for each new Kaggle submission (competition day 14 Oct).

Features (all from X only — no target leakage):
  Base: cohort, sexM, gene, age_at_diagnosis, age, ledd, timing, rater_id, on, off
  Patient aggregates (05): mean/std/median of off/on/ledd per patient,
    visit count, age span, slope of off/on over age, prev/next visit values
  Clinical/missing (06): disease duration, binary missing indicators,
    visit_type (both/off_only/on_only), on_off_diff, ratios
  Rater (08): rater's mean off/on from train (rater systematic bias)

Post-processing:
  Deg-2 polynomial projection per patient: smooths LightGBM predictions
  along the patient's age axis (gain +0.145 RMSE, paired std 0.017).
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
REPORT_KEY   = "16_final"
PROJECT_NAME = "16-final"
CLIP_MIN, CLIP_MAX = 0, 132

PROJECT_ROOT = Path(__file__).resolve().parents[1]

skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

# ── Data ──────────────────────────────────────────────────────────────────────
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
    """Patient-level aggregates from X_df only — no target, no leakage."""
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
    rf = df[["patient_id", "age", "pt_visit_rank",
             "pt_off_prev", "pt_off_next",
             "pt_on_prev",  "pt_on_next"]].copy()

    out = X_df.reset_index().merge(agg, on="patient_id", how="left")
    out = out.merge(rf, on=["patient_id", "age"], how="left")
    return out.set_index("Index")


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Disease duration, missing indicators, visit type, on/off ratios."""
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
    """Rater mean off/on from train split — rater systematic bias."""
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
    src = X_df_source if X_df_source is not None else X_df_target
    tgt = add_patient_features(X_df_target)
    tgt = add_clinical_features(tgt)
    tgt = add_rater_features(tgt, src)
    pt_cols  = [c for c in tgt.columns if c.startswith("pt_")]
    return tgt[BASE_FEATURES + pt_cols + CLIN_COLS + RATER_COLS]


def as_lgbm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert str/object columns to 'category' for LightGBM native handling."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


def make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        random_state=0, n_estimators=1000, learning_rate=0.05,
        num_leaves=127, min_child_samples=20, verbose=-1,
    )


# ── Polynomial projection post-processing ─────────────────────────────────────

def poly_project(age_arr: np.ndarray, pred_arr: np.ndarray,
                 degree: int = 2, alpha: float = 1.0) -> np.ndarray:
    """
    Ridge-regularized polynomial fit of (age → pred) for one patient.
    Returns smoothed predictions. alpha regularizes non-intercept coefficients.
    Falls back to lower degree when n_visits < degree + 1.
    """
    n = len(age_arr)
    eff_deg = min(degree, n - 1)
    if eff_deg == 0:
        return np.full(n, pred_arr.mean())
    age_c  = age_arr - age_arr.mean()
    age_sc = age_c / (age_c.std() + 1e-8)
    V = np.column_stack([age_sc ** d for d in range(eff_deg + 1)])
    reg = alpha * np.eye(V.shape[1])
    reg[0, 0] = 0.0  # don't penalise intercept
    coef = np.linalg.solve(V.T @ V + reg, V.T @ pred_arr)
    return V @ coef


def project_patient_preds(pred_series: pd.Series, X_meta: pd.DataFrame,
                          degree: int = 2, alpha: float = 1.0) -> pd.Series:
    """Apply poly_project per patient; return smoothed predictions."""
    result = pred_series.copy()
    for pid, grp in X_meta.groupby("patient_id"):
        idx  = grp.index
        age  = grp["age"].values
        smoothed = poly_project(age, pred_series.loc[idx].values, degree, alpha)
        result.loc[idx] = smoothed
    return result


# ── Pre-compute GroupKFold-5 folds ────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds_raw = list(gkf.split(X_raw, y, groups=groups))

# ── CV evaluation (OOF + projection) ─────────────────────────────────────────
print(f"[{REPORT_KEY}] GroupKFold-5 OOF + deg-2 projection...")
t0 = time.time()

fold_rmse_raw  = []
fold_rmse_proj = []
oof_proj = np.full(len(y), np.nan)

for fold_idx, (train_idx, val_idx) in enumerate(folds_raw):
    X_tr_raw  = X_raw.iloc[train_idx]
    X_val_raw = X_raw.iloc[val_idx]
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    X_tr  = as_lgbm_df(build_features(X_tr_raw,  X_df_source=X_tr_raw))
    X_val = as_lgbm_df(build_features(X_val_raw, X_df_source=X_tr_raw))

    model = make_lgbm()
    model.fit(X_tr, y_tr)
    preds_raw = model.predict(X_val)

    preds_proj = project_patient_preds(
        pd.Series(preds_raw, index=X_val_raw.index),
        X_val_raw, degree=2, alpha=1.0
    ).values

    oof_proj[val_idx] = preds_proj

    rmse_raw  = float(np.sqrt(np.mean((y_val.values - preds_raw)  ** 2)))
    rmse_proj = float(np.sqrt(np.mean((y_val.values - preds_proj) ** 2)))
    fold_rmse_raw.append(rmse_raw)
    fold_rmse_proj.append(rmse_proj)
    print(f"  Fold {fold_idx+1}  raw={rmse_raw:.4f}  proj={rmse_proj:.4f}  "
          f"gain={rmse_raw - rmse_proj:+.4f}")

cv_elapsed = time.time() - t0
mean_raw   = float(np.mean(fold_rmse_raw))
mean_proj  = float(np.mean(fold_rmse_proj))
std_proj   = float(np.std(fold_rmse_proj))
paired_gain = float(np.mean([fold_rmse_raw[i] - fold_rmse_proj[i] for i in range(5)]))
paired_std  = float(np.std([fold_rmse_raw[i] - fold_rmse_proj[i] for i in range(5)]))

print(f"\n  LightGBM raw:       {mean_raw:.4f} +/- {float(np.std(fold_rmse_raw)):.4f}")
print(f"  + deg-2 projection: {mean_proj:.4f} +/- {std_proj:.4f}  "
      f"(paired gain {paired_gain:.4f} +/- {paired_std:.4f})")
print(f"  Previous best (11_final): 4.3705 +/- 0.0942")
print(f"  CV time: {cv_elapsed:.0f}s")

# ── Push CV report ────────────────────────────────────────────────────────────
# The hub report uses LightGBM CV on full-train features (no projection —
# skore.evaluate doesn't support post-proc; the honest OOF RMSE is above).
X_full     = as_lgbm_df(build_features(X_raw, X_df_source=X_raw))
folds_full = list(gkf.split(X_full, y, groups=groups))
skore_model = make_lgbm()
report_cv   = skore.evaluate(skore_model, X_full, y,
                              splitter=folds_full, n_jobs=-1)

skore.login()
proj_hub = skore.Project(PROJECT_NAME, mode="hub", workspace="bob")
proj_hub.put(REPORT_KEY, report_cv)
print(f"\nHub report pushed as '{REPORT_KEY}'")
print(f"Hub URL: https://skore.probabl.ai/bob/{PROJECT_NAME}/cross-validations/")
print("(Paste the full URL printed above into the Kaggle Submission Description)")

# ── Refit on all training data ────────────────────────────────────────────────
print("\nRefitting on all training data...")
t_refit = time.time()
final_model = make_lgbm()
final_model.fit(X_full, y)
print(f"  Refit in {time.time() - t_refit:.0f}s")

# ── Test predictions + polynomial projection ──────────────────────────────────
X_test_full = as_lgbm_df(build_features(X_test_raw, X_df_source=X_raw))
test_preds_raw = final_model.predict(X_test_full)

test_preds_proj = project_patient_preds(
    pd.Series(test_preds_raw, index=X_test_raw.index),
    X_test_raw, degree=2, alpha=1.0
).values

test_preds = np.clip(test_preds_proj, CLIP_MIN, CLIP_MAX)

assert int(np.isnan(test_preds).sum()) == 0, "NaN in test predictions"
assert len(test_preds) == len(X_test_raw), "Row count mismatch"

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path = submissions_dir / f"{REPORT_KEY}.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_preds}).to_csv(
    sub_path, index=False
)

total_elapsed = time.time() - t0
print(f"\nSubmission: {sub_path}  ({len(X_test_raw)} rows, NaN=0)")
print("\n" + "=" * 70)
print(f"FINAL SUMMARY — {REPORT_KEY}")
print("=" * 70)
print(f"  GroupKFold-5 RMSE (with projection): {mean_proj:.4f} +/- {std_proj:.4f}")
print(f"  vs 11_final (no projection):          4.3705 +/- 0.0942")
print(f"  Improvement:                         +{4.3705 - mean_proj:.4f} (paired {paired_gain:.4f} +/- {paired_std:.4f})")
print(f"  Total time: {total_elapsed:.0f}s")
print(f"  Submission: {sub_path}")
print(f"  Hub key:    {REPORT_KEY}")
print("=" * 70)
