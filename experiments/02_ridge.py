# %% [markdown]
# # 02 Ridge Regression — Holdout Baseline
#
# Ridge on numeric clinical features with median imputation.
# Same 80/20 holdout as 01_dummy so RMSE is directly comparable.
#
# Features: age_at_diagnosis, age, ledd, time_since_intake_on,
#            time_since_intake_off, on, off, sexM
# Excluded: patient_id (identifier), cohort/gene/rater_id (string categoricals)

# %%
import json
import os
from pathlib import Path

import pandas as pd
import skore
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Auth ─────────────────────────────────────────────────────────────────────
skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

# ── Data ─────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")

NUMERIC_FEATURES = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = X_train[NUMERIC_FEATURES]
y = y_train["target"]

# ── Pipeline ─────────────────────────────────────────────────────────────────
ridge_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# ── Evaluate — same 80/20 holdout as 01_dummy ───────────────────────────────
report = skore.evaluate(ridge_pipe, X, y, splitter=0.2)
print(report.metrics.rmse())

# ── Push to hub ───────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("02-ridge", mode="hub", workspace="bob")
project.put("02_ridge_holdout", report)
print("Pushed: 02_ridge_holdout")
