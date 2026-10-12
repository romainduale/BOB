# %% [markdown]
# # Experiment 04 — Longitudinal HistGBR Final Submission
#
# **Date:** 2026-10-08
# **Goal:** Fit the exact longitudinal HistGBR pipeline on ALL training data
# and generate test predictions.
#
# CV result (from Exp-03 probe):
#   Fold RMSEs : [4.5388, 4.6301, 4.5255, 4.4089, 4.5358]  Mean: 4.5278
#
# Output: submissions/04_longitudinal_histgbr.csv
# Skore Hub key: 04_longitudinal_histgbr

# %%
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skore

from parkinsons import PROJECT_ROOT
from parkinsons.hub import load_skore_credentials

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

groups = X_train_raw["patient_id"].values

# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (identical to Exp-03 probe)
# ──────────────────────────────────────────────────────────────────────────────
NULLABLE   = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]
SLOPE_COLS = ["on", "off", "ledd"]
CAT_COLS   = ["cohort", "gene", "rater_id", "sexM"]


def _patient_slope_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Within-patient OLS of `col` ~ age → slope and fitted value per visit."""
    slope_out  = np.full(len(df), np.nan)
    fitted_out = np.full(len(df), np.nan)

    for pid, grp in df.groupby("patient_id"):
        valid = grp[[col, "age"]].dropna()
        idx   = grp.index
        pos   = df.index.get_indexer(idx)
        if len(valid) >= 2:
            res = linregress(valid["age"].values, valid[col].values)
            slope_out[pos]  = res.slope
            fitted_out[pos] = res.intercept + res.slope * grp.loc[idx, "age"].values
        else:
            slope_out[pos]  = np.nan
            fitted_out[pos] = grp.loc[idx, col].values

    return pd.DataFrame(
        {f"{col}_pat_slope": slope_out, f"{col}_pat_fitted": fitted_out},
        index=df.index,
    )


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full longitudinal feature frame from a raw visit frame."""
    df = df.copy()

    # row-level
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"]     = df["off"] - df["on"]
    for col in NULLABLE:
        df[f"{col}_missing"] = df[col].isna().astype(float)

    # patient-level age stats + visit count
    pat = df.groupby("patient_id")
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
                f"{col}_pat_mean":     "mean",
                f"{col}_pat_median":   "median",
                f"{col}_pat_min":      "min",
                f"{col}_pat_max":      "max",
                f"{col}_pat_std":      "std",
                f"{col}_pat_count":    lambda s: s.notna().sum(),
                f"{col}_pat_obs_frac": lambda s: s.notna().mean(),
            }
        )
        df = df.join(col_stats, on="patient_id")

    # within-patient slope + fitted value for on, off, ledd
    for col in SLOPE_COLS:
        df = df.join(_patient_slope_features(df, col))

    # encode categoricals as integer codes (HistGBR handles natively)
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col]).codes

    return df.drop(columns=["patient_id"])


# ──────────────────────────────────────────────────────────────────────────────
# 3. BUILD TRAIN AND TEST FEATURE FRAMES
#    Test patient aggregates are computed from X_test only.
# ──────────────────────────────────────────────────────────────────────────────
print("Engineering train features …", flush=True)
X_feat_train = engineer(X_train_raw)

print("Engineering test features …", flush=True)
X_feat_test  = engineer(X_test_raw)

print(f"Train: {X_feat_train.shape}  Test: {X_feat_test.shape}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4. FIT ON ALL TRAINING DATA
# ──────────────────────────────────────────────────────────────────────────────
MODEL = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
    categorical_features="from_dtype",
)

print("Fitting on all training data …", flush=True)
MODEL.fit(X_feat_train, y_train)

# ──────────────────────────────────────────────────────────────────────────────
# 5. PREDICT ON TEST — preserve original Index ordering
# ──────────────────────────────────────────────────────────────────────────────
test_preds = MODEL.predict(X_feat_test)

submission = pd.DataFrame(
    {"target": test_preds},
    index=X_test_raw.index,   # original Index, order preserved
)
submission.index.name = "Index"

# ──────────────────────────────────────────────────────────────────────────────
# 6. WRITE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────────
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
sub_path = SUBMISSIONS_DIR / "04_longitudinal_histgbr.csv"
submission.to_csv(sub_path)

# ──────────────────────────────────────────────────────────────────────────────
# 7. SKORE REPORT
#    Re-run the 5-fold CV with the same groups/splitter to get OOF predictions
#    for the EstimatorReport, then push to Hub.
# ──────────────────────────────────────────────────────────────────────────────
print("Running CV for Skore OOF report …", flush=True)
FOLDS = list(GroupKFold(n_splits=5).split(X_feat_train, y_train, groups=groups))

oof_preds = np.full(len(y_train), np.nan)
for tr, va in FOLDS:
    m = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        categorical_features="from_dtype",
    )
    m.fit(X_feat_train.iloc[tr], y_train[tr])
    oof_preds[va] = m.predict(X_feat_train.iloc[va])

oof_rmse = float(np.sqrt(np.mean((oof_preds - y_train) ** 2)))
print(f"OOF RMSE (global): {oof_rmse:.4f}", flush=True)

# EstimatorReport API rules:
#   - already-fitted estimator → X_test/y_test only (no train data allowed)
#   - unfitted estimator       → X_train/y_train + X_test/y_test required
#
# Strategy: pass an unfitted clone; use fold-1 train/val split so skore
# fits on the fold-1 training rows and evaluates on the fold-1 val rows.
# This gives an honest held-out regression report without re-running all CV.
tr0, va0 = FOLDS[0]
report = skore.EstimatorReport(
    HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        categorical_features="from_dtype",
    ),
    X_train=X_feat_train.iloc[tr0],
    y_train=y_train[tr0],
    X_test=X_feat_train.iloc[va0],
    y_test=y_train[va0],
)

load_skore_credentials()
skore.login(mode="hub")
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
project = skore.Project(
    "04_longitudinal_histgbr",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("04_longitudinal_histgbr", report)

report_url = (
    f"{cfg['hub_url']}/workspaces/{cfg['workspace']}"
    f"/projects/04_longitudinal_histgbr"
)

# ──────────────────────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSubmission path : {sub_path}")
print(f"Row count       : {len(submission)}")
print(f"Prediction min  : {test_preds.min():.4f}")
print(f"Prediction mean : {test_preds.mean():.4f}")
print(f"Prediction max  : {test_preds.max():.4f}")
print(f"Skore Hub URL   : {report_url}")
