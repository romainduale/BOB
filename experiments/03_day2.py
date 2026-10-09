# %% [markdown]
# # 03 Day 2 — Patient-Grouped Cross-Validation
#
# Fixed 5-fold GroupKFold on patient_id — same folds for every model so
# RMSE values are directly comparable.
#
# Models evaluated:
#   A. DummyRegressor(strategy="mean")           — grouped-CV lower bound
#   B. Ridge(alpha=1.0) + SimpleImputer(median)  — numeric features only
#   C. HistGradientBoostingRegressor              — numeric features, NaN preserved
#   D. TableVectorizer + HGBR                    — all features incl. categoricals
#
# Best model is refitted on all training data → Kaggle submission.

# %%
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Auth ─────────────────────────────────────────────────────────────────────
skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]

# ── Data ─────────────────────────────────────────────────────────────────────
X_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_df = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

y = y_df["target"]
groups = X_raw["patient_id"]

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

ALL_FEATURES = [
    "cohort",
    "sexM",
    "gene",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "rater_id",
    "on",
    "off",
]

X_num = X_raw[NUMERIC_FEATURES]
X_all = X_raw[ALL_FEATURES]
X_test_num = X_test_raw[NUMERIC_FEATURES]
X_test_all = X_test_raw[ALL_FEATURES]

# ── Pre-compute fixed folds (groups-aware) ────────────────────────────────────
# Convert to integer positional indices (skore CV uses iloc-style indexing).
gkf = GroupKFold(n_splits=5)
folds_num = list(gkf.split(X_num, y, groups=groups))
folds_all = list(gkf.split(X_all, y, groups=groups))

# ── Pipelines ─────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")

ridge_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# HGBR handles NaN natively — no imputer
hgbr_pipe = HistGradientBoostingRegressor(random_state=0)

skrub_pipe = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)

# ── Evaluate all models on the same grouped folds ─────────────────────────────
print("Evaluating DummyRegressor (grouped CV)…")
report_dummy_gcv = skore.evaluate(
    dummy, X_num, y, splitter=iter(folds_num), n_jobs=-1,
)

print("Evaluating Ridge (grouped CV)…")
report_ridge_gcv = skore.evaluate(
    ridge_pipe, X_num, y, splitter=iter(folds_num), n_jobs=-1,
)

print("Evaluating HGBR numeric (grouped CV)…")
report_hgbr_gcv = skore.evaluate(
    hgbr_pipe, X_num, y, splitter=iter(folds_num), n_jobs=-1,
)

print("Evaluating Skrub TableVectorizer+HGBR (grouped CV)…")
report_skrub_gcv = skore.evaluate(
    skrub_pipe, X_all, y, splitter=iter(folds_all), n_jobs=-1,
)

# ── Collect RMSE stats ────────────────────────────────────────────────────────
def rmse_stats(report, label):
    try:
        vals = report.metrics.rmse().to_numpy().ravel()
        return {"model": label, "mean_rmse": float(np.mean(vals)), "std_rmse": float(np.std(vals))}
    except Exception:
        v = float(report.metrics.rmse())
        return {"model": label, "mean_rmse": v, "std_rmse": float("nan")}

results = [
    rmse_stats(report_dummy_gcv,  "DummyRegressor (grouped CV)"),
    rmse_stats(report_ridge_gcv,  "Ridge (grouped CV)"),
    rmse_stats(report_hgbr_gcv,   "HGBR numeric (grouped CV)"),
    rmse_stats(report_skrub_gcv,  "Skrub HGBR (grouped CV)"),
]

for r in results:
    print(f"  {r['model']:<45}  RMSE {r['mean_rmse']:.4f} ± {r['std_rmse']:.4f}")

# ── Push reports to hub ───────────────────────────────────────────────────────
skore.login()

for proj_name, key, report in [
    ("03-day2-dummy-gcv",  "03_dummy_grouped_cv",  report_dummy_gcv),
    ("03-day2-ridge-gcv",  "03_ridge_grouped_cv",  report_ridge_gcv),
    ("03-day2-hgbr-gcv",   "03_hgbr_grouped_cv",   report_hgbr_gcv),
    ("03-day2-skrub-gcv",  "03_skrub_grouped_cv",  report_skrub_gcv),
]:
    proj = skore.Project(proj_name, mode="hub", workspace="bob")
    proj.put(key, report)
    print(f"Pushed: {key}")

# ── Best model ────────────────────────────────────────────────────────────────
best = min(results, key=lambda r: r["mean_rmse"])
print(f"\nBest: {best['model']}  RMSE={best['mean_rmse']:.4f}")

label_to_config = {
    "Skrub HGBR (grouped CV)":      (skrub_pipe, X_all, X_test_all, "03_skrub_grouped_cv"),
    "HGBR numeric (grouped CV)":    (hgbr_pipe,  X_num, X_test_num, "03_hgbr_grouped_cv"),
    "Ridge (grouped CV)":           (ridge_pipe, X_num, X_test_num, "03_ridge_grouped_cv"),
    "DummyRegressor (grouped CV)":  (dummy,      X_num, X_test_num, "03_dummy_grouped_cv"),
}
best_pipe, X_fit, X_pred, report_key = label_to_config[best["model"]]

print(f"Refitting {best['model']} on all {len(X_fit)} rows…")
best_pipe.fit(X_fit, y)
preds = best_pipe.predict(X_pred)

# ── Kaggle submission ─────────────────────────────────────────────────────────
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission_path = submissions_dir / f"{report_key}.csv"
submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds})
submission.to_csv(submission_path, index=False)
print(f"Kaggle submission: {submission_path}  ({len(submission)} rows)")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)
print(f"{'Model':<45} {'RMSE mean':>10} {'RMSE std':>10} {'Δ vs 16.48':>12}  Report key")
print("-" * 90)
for r in results:
    delta = 16.48 - r["mean_rmse"]
    key_slug = {
        "DummyRegressor (grouped CV)": "03_dummy_grouped_cv",
        "Ridge (grouped CV)":          "03_ridge_grouped_cv",
        "HGBR numeric (grouped CV)":   "03_hgbr_grouped_cv",
        "Skrub HGBR (grouped CV)":     "03_skrub_grouped_cv",
    }[r["model"]]
    marker = " ◀ BEST" if r["model"] == best["model"] else ""
    print(f"{r['model']:<45} {r['mean_rmse']:>10.4f} {r['std_rmse']:>10.4f} {delta:>+12.4f}  {key_slug}{marker}")
print("-" * 90)
print(f"\nBest model (grouped-CV): {best['model']}")
print(f"Grouped-CV RMSE:         {best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}")
print(f"Skore report key:        {report_key}")
print(f"Kaggle submission path:  {submission_path}")
