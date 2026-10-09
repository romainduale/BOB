# %% [markdown]
# # Experiment 01 — Baselines
#
# **Date:** 2026-10-08
# **Goal:** Establish four leakage-safe RMSE baselines under identical
# `GroupKFold(n_splits=5)` folds with `groups=patient_id`.
# **Result:** filled in after the run.
#
# **Fit budget:**
#   - Dummy × 5 folds            =  5 fits
#   - OffProxy × 5 folds         =  5 fits  (trivial)
#   - Ridge × 5 folds            =  5 fits
#   - HistGBR via skore.evaluate =  5 fits  (skore owns these; no cv_rmse duplicate)
#   - HistGBR final refit        =  1 fit
#   Total                        = 21 fits
#
# `n_jobs=1` on `skore.evaluate` keeps joblib sequential (no loky worker
# spawning), which avoids a Windows dead-lock inside IPython's run_cell.
# HistGradientBoostingRegressor has no n_jobs parameter; it uses OpenMP.

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from parkinsons import PROJECT_ROOT
from parkinsons.hub import load_skore_credentials

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Load data

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

# Groups for GroupKFold — NOT used as a predictive feature
groups = X_train_raw["patient_id"].values

# %% [markdown]
# ## Feature engineering
#
# All transformations reference only columns present in X_train/X_test.
# Missing indicators are added for the five high-missingness columns.
# `patient_id` and `cohort` are excluded from features; `rater_id` kept
# as a high-cardinality categorical for tree models, dropped for Ridge
# (too many dummies — handled separately).

# %%
NUMERIC_COLS = [
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CATEGORICAL_COLS = ["sexM", "gene", "cohort", "rater_id"]
MISSING_INDICATOR_COLS = ["off", "on", "ledd", "time_since_intake_on", "time_since_intake_off"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add disease_duration, off_minus_on, and missing indicators.

    Parameters
    ----------
    df :
        Raw feature dataframe (X_train or X_test), without patient_id.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered columns appended. Original columns
        are preserved so downstream imputers/encoders work normally.
    """
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"] = df["off"] - df["on"]
    for col in MISSING_INDICATOR_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(float)
    return df


def drop_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """Remove patient_id from feature matrix."""
    return df.drop(columns=["patient_id"], errors="ignore")


# Apply feature engineering (before splitting — pure column ops, no target info)
X_train_eng = engineer_features(drop_group_id(X_train_raw))
X_test_eng = engineer_features(drop_group_id(X_test_raw))

# Updated column lists after engineering
NUMERIC_COLS_ENG = NUMERIC_COLS + ["disease_duration", "off_minus_on"]
INDICATOR_COLS_ENG = [f"{c}_missing" for c in MISSING_INDICATOR_COLS]
# Ridge: drop rater_id (50 dummies inflates complexity; use other categoricals)
CATEGORICAL_COLS_RIDGE = ["sexM", "gene", "cohort"]
# HistGBR: include rater_id via OrdinalEncoder
CATEGORICAL_COLS_HIST = ["sexM", "gene", "cohort", "rater_id"]

# %% [markdown]
# ## Cross-validation setup
#
# One `GroupKFold` object; pre-compute folds once so every model sees
# identical train/val splits. `patient_id` is used only here.

# %%
CV = GroupKFold(n_splits=5)
FOLDS = list(CV.split(X_train_eng, y_train, groups=groups))
N_FOLDS = len(FOLDS)

# %% [markdown]
# ## Helper: fold-level RMSE

# %%

def cv_rmse(estimator, X: pd.DataFrame, y: np.ndarray, folds: list) -> dict:
    """Compute per-fold RMSE and summary statistics.

    Parameters
    ----------
    estimator :
        Sklearn-compatible estimator with fit/predict.
    X :
        Full training feature matrix (engineered, no patient_id).
    y :
        Training targets.
    folds :
        Pre-computed list of (train_idx, val_idx) tuples.

    Returns
    -------
    dict
        Keys: fold_rmses, mean, std.
    """
    rmses = []
    for fold_i, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"  fold {fold_i}/{len(folds)}...", flush=True)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        estimator.fit(X_tr, y_tr)
        preds = estimator.predict(X_val)
        rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
        rmses.append(rmse)
    return {
        "fold_rmses": [round(r, 4) for r in rmses],
        "mean": round(float(np.mean(rmses)), 4),
        "std": round(float(np.std(rmses)), 4),
    }


# %% [markdown]
# ## Model 1 — DummyRegressor (mean)
#
# Predicts the training-fold target mean for every row. Establishes the
# RMSE floor.

# %%
print("Model 1/4: DummyRegressor...")
dummy = DummyRegressor(strategy="mean")
results_dummy = cv_rmse(dummy, X_train_eng, y_train, FOLDS)
print("Dummy:", results_dummy)

# %% [markdown]
# ## Model 2 — off-proxy baseline
#
# When `off` is available, predict `off` directly.
# When `off` is missing, fall back to the training-fold target mean.
# Evaluated inside the same GroupKFold folds; the training-fold mean
# is computed from y_train only (no leakage).

# %%

class OffProxyRegressor:
    """Predict off when observed; otherwise predict training-fold target mean.

    All methods follow the sklearn estimator interface so cv_rmse can
    use it transparently.
    """

    def __init__(self):
        self.train_mean_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.train_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        off_vals = X["off"].values.copy()
        missing = np.isnan(off_vals)
        off_vals[missing] = self.train_mean_
        return off_vals


print("Model 2/4: OffProxyRegressor...")
results_off = cv_rmse(OffProxyRegressor(), X_train_eng, y_train, FOLDS)
print("Off-proxy:", results_off)

# %% [markdown]
# ## Model 3 — Ridge regression
#
# Numerical features: median imputation + standard scaling.
# Categorical features: constant ("missing") imputation + OneHotEncoder.
# Missing indicators passed through as-is (already 0/1 floats).
# `rater_id` excluded (50 categories → 50 dummies; keeps pipeline lean).

# %%
numeric_transformer_ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer_ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor_ridge = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_ridge, NUMERIC_COLS_ENG),
        ("cat", categorical_transformer_ridge, CATEGORICAL_COLS_RIDGE),
        ("ind", "passthrough", INDICATOR_COLS_ENG),
    ],
    remainder="drop",
)

ridge_pipeline = Pipeline([
    ("preprocessor", preprocessor_ridge),
    ("model", Ridge(alpha=10)),
])

print("Model 3/4: Ridge...")
results_ridge = cv_rmse(ridge_pipeline, X_train_eng, y_train, FOLDS)
print("Ridge:", results_ridge)

# %% [markdown]
# ## Model 4 — HistGradientBoostingRegressor + Skore report
#
# Categorical columns encoded with OrdinalEncoder (HistGBR requires
# integer categories). Numeric features passed with NaN left in place —
# HistGBR handles them natively. Missing indicators included as extra
# numeric features.
#
# HistGBR CV is run **once** via `skore.evaluate` (not via `cv_rmse`)
# to avoid running the same 5 folds twice. The fold indices come from
# the pre-computed `FOLDS` list, ensuring comparability with the other
# three models. `n_jobs=1` on `skore.evaluate` keeps joblib sequential,
# preventing the loky worker dead-lock on Windows inside IPython's run_cell.

# %%
categorical_transformer_hist = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor_hist = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_hist, CATEGORICAL_COLS_HIST),
        ("num", "passthrough", NUMERIC_COLS_ENG),
        ("ind", "passthrough", INDICATOR_COLS_ENG),
    ],
    remainder="drop",
)

hist_pipeline = Pipeline([
    ("preprocessor", preprocessor_hist),
    ("model", HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
    )),
])

# %%
print("Model 4/4: HistGBR — running via skore.evaluate (5 folds, n_jobs=1)...")
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="ibm-hackaton",
    mode="hub",
    workspace=cfg["workspace"],
)

report = skore.evaluate(
    hist_pipeline,
    X=X_train_eng,
    y=y_train,
    splitter=iter(FOLDS),
    n_jobs=1,
)
project.put("01_baselines_hist", report)
print("HistGBR report stored.")

# Extract per-fold RMSE from each sub-report (one per fold)
hist_fold_rmses = [float(r.metrics.rmse()) for r in report.reports_]
hist_mean = float(np.mean(hist_fold_rmses))
hist_std = float(np.std(hist_fold_rmses))
results_hist = {
    "fold_rmses": [round(v, 4) for v in hist_fold_rmses],
    "mean": round(hist_mean, 4),
    "std": round(hist_std, 4),
}
print("HistGBR:", results_hist)

# %% [markdown]
# ## Comparison table

# %%
comparison = pd.DataFrame([
    {"Model": "DummyRegressor(mean)", **results_dummy},
    {"Model": "OffProxyRegressor",    **results_off},
    {"Model": "Ridge(alpha=10)",      **results_ridge},
    {"Model": "HistGBR(n_jobs=1)",    **results_hist},
]).sort_values("mean").reset_index(drop=True)

print("\nCV RMSE comparison (sorted by mean RMSE):")
print(comparison[["Model", "mean", "std", "fold_rmses"]].to_string(index=False))

# %% [markdown]
# ## Kaggle submission — HistGBR refit on all training data

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final_model = Pipeline([
    ("preprocessor", preprocessor_hist),
    ("model", HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
    )),
])
final_model.fit(X_train_eng, y_train)
test_preds = final_model.predict(X_test_eng)

submission = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds})
submission.to_csv(submissions_dir / "01_baselines.csv", index=False)
print(f"Submission written: {submissions_dir / '01_baselines.csv'} ({len(submission)} rows)")
