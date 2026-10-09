# %% [markdown]
# # Experiment 03 — Longitudinal HistGBR (compact patient-level features)
#
# **Date:** 2026-10-08
# **Goal:** Add rich patient-level longitudinal features to HistGBR and
# compare against the Exp-02 CatBoost baseline.
#
# No Skore upload. No tuning. 5-fold GroupKFold RMSE only.

# %%
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values

groups = X_raw["patient_id"].values

# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
NULLABLE = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]
SLOPE_COLS = ["on", "off", "ledd"]   # also get within-patient regression


def _patient_slope_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    For each patient compute within-patient OLS of `col` ~ age.
    Returns a frame with columns:
      {col}_pat_slope, {col}_pat_fitted
    indexed like df.
    """
    slope_out   = np.full(len(df), np.nan)
    fitted_out  = np.full(len(df), np.nan)

    for pid, grp in df.groupby("patient_id"):
        valid = grp[[col, "age"]].dropna()
        idx   = grp.index
        if len(valid) >= 2:
            res    = linregress(valid["age"].values, valid[col].values)
            slope  = res.slope
            fitted = res.intercept + res.slope * grp.loc[idx, "age"].values
        else:
            slope  = np.nan
            fitted = grp.loc[idx, col].values  # fallback: just the observed value

        pos = df.index.get_indexer(idx)
        slope_out[pos]  = slope
        fitted_out[pos] = fitted

    return pd.DataFrame(
        {f"{col}_pat_slope": slope_out, f"{col}_pat_fitted": fitted_out},
        index=df.index,
    )


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── row-level numerics ──
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"]     = df["off"] - df["on"]
    for col in NULLABLE:
        df[f"{col}_missing"] = df[col].isna().astype(float)

    # ── patient-level aggregates ──
    pat = df.groupby("patient_id")

    # age stats
    age_stats = pat["age"].agg(
        patient_age_mean="mean",
        patient_age_min="min",
        patient_age_max="max",
    )
    age_stats["patient_age_span"] = age_stats["patient_age_max"] - age_stats["patient_age_min"]
    n_visits = pat.size().rename("n_visits")

    df = df.join(age_stats, on="patient_id").join(n_visits, on="patient_id")
    df["age_rel_mean"] = df["age"] - df["patient_age_mean"]
    df["age_rel_min"]  = df["age"] - df["patient_age_min"]

    # per-column stats for NULLABLE
    for col in NULLABLE:
        col_stats = pat[col].agg(
            **{
                f"{col}_pat_mean":   "mean",
                f"{col}_pat_median": "median",
                f"{col}_pat_min":    "min",
                f"{col}_pat_max":    "max",
                f"{col}_pat_std":    "std",
                f"{col}_pat_count":  lambda s: s.notna().sum(),
                f"{col}_pat_obs_frac": lambda s: s.notna().mean(),
            }
        )
        df = df.join(col_stats, on="patient_id")

    # within-patient slope + fitted value for on, off, ledd
    for col in SLOPE_COLS:
        df = df.join(_patient_slope_features(df, col))

    return df


print("Engineering features …", flush=True)
X_eng = engineer(X_raw)
print(f"Feature frame: {X_eng.shape}", flush=True)

# ── encode categoricals as integers (HistGBR handles them natively) ──
CAT_COLS = ["cohort", "gene", "rater_id", "sexM"]
for col in CAT_COLS:
    X_eng[col] = pd.Categorical(X_eng[col]).codes  # -1 for NaN → treated as missing

# drop patient_id — never a predictive feature
X_feat = X_eng.drop(columns=["patient_id"])
print(f"Final feature count: {X_feat.shape[1]}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
FOLDS = list(GroupKFold(n_splits=5).split(X_feat, y, groups=groups))

MODEL = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
    categorical_features="from_dtype",
)

fold_rmses = []
for i, (tr, va) in enumerate(FOLDS, 1):
    MODEL.fit(X_feat.iloc[tr], y[tr])
    pred = MODEL.predict(X_feat.iloc[va])
    rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
    fold_rmses.append(rmse)
    print(f"Fold {i}/5  RMSE={rmse:.4f}", flush=True)

mean_rmse = float(np.mean(fold_rmses))
print(f"\nFold RMSEs : {[round(r, 4) for r in fold_rmses]}")
print(f"Mean RMSE  : {mean_rmse:.4f}")
