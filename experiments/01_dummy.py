# %% [markdown]
# # 01 Dummy Regressor — Baseline
#
# Establish a lower bound on performance by predicting the mean of `y_train`
# for every row. Any useful model must beat these numbers.
#
# - **Splitter:** 0.2 holdout (single 80/20 train-test split)
# - **Estimator:** `DummyRegressor(strategy="mean")`
# - **Report name:** `01_dummy` pushed to skore hub workspace `bobathon-esilv`

# %%
import json
import os
from pathlib import Path

import pandas as pd
import skore
from sklearn.dummy import DummyRegressor

# Locate repo root from this file's position (experiments/ is one level down).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Load API key from .skore ────────────────────────────────────────────────
skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

# ── Load data ────────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

X = X_train
y = y_train["target"]

# ── Evaluate ─────────────────────────────────────────────────────────────────
# splitter=0.2 → single 80/20 holdout → EstimatorReport
estimator = DummyRegressor(strategy="mean")
report = skore.evaluate(estimator, X, y, splitter=0.2)

# ── Push to hub ──────────────────────────────────────────────────────────────
skore.login()
project = skore.Project("01_dummy", mode="hub", workspace="bob")
project.put("01_dummy", report)

# ── Kaggle submission ─────────────────────────────────────────────────────────
estimator.fit(X, y)
submission = pd.DataFrame({"target": estimator.predict(X_test)}, index=X_test.index)
submission.index.name = "Index"
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "01_dummy.csv")
print(f"Submission written: submissions/01_dummy.csv  ({len(submission)} rows)")
print("Done — report '01_dummy' pushed to hub workspace 'bob'.")
