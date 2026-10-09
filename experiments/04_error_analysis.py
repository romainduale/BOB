# %% [markdown]
# # 04 Day 3 — Error Analysis: Skrub TableVectorizer + HGBR
#
# Diagnostic only. Same 5-fold GroupKFold(patient_id) splits as 03_day2.py.
# No model changes, no feature engineering, no hyperparameter tuning.
#
# Outputs (all saved to results/day3_diagnostics/):
#   fold_stats.csv           — RMSE, MAE, rows, unique patients, target stats per fold
#   oof_predictions.csv      — out-of-fold predictions for every training row
#   rmse_by_patient.csv      — RMSE/MAE/n_obs per patient_id
#   rmse_by_cohort.csv       — RMSE/MAE/n_obs per cohort
#   rmse_by_gene.csv         — RMSE/MAE/n_obs per gene
#   rmse_by_rater.csv        — RMSE/MAE/n_obs per rater_id
#   top_errors.csv           — 20 rows with largest absolute prediction error

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "day3_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
X_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_df  = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
y = y_df["target"]
groups = X_raw["patient_id"]

ALL_FEATURES = [
    "cohort", "sexM", "gene", "age_at_diagnosis", "age",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "rater_id", "on", "off",
]
X = X_raw[ALL_FEATURES]

# ── Exact same folds as 03_day2.py ────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups=groups))

# ── OOF loop ─────────────────────────────────────────────────────────────────
oof_pred = np.full(len(y), np.nan)
fold_records = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipe = make_pipeline(
        TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)
    oof_pred[val_idx] = preds

    residuals = y_val.values - preds
    fold_records.append({
        "fold":           fold_idx + 1,
        "n_rows":         len(val_idx),
        "n_unique_patients": groups.iloc[val_idx].nunique(),
        "rmse":           float(np.sqrt(np.mean(residuals ** 2))),
        "mae":            float(np.mean(np.abs(residuals))),
        "target_mean":    float(y_val.mean()),
        "target_std":     float(y_val.std()),
        "target_min":     float(y_val.min()),
        "target_max":     float(y_val.max()),
    })
    print(f"  Fold {fold_idx + 1}  RMSE={fold_records[-1]['rmse']:.4f}  "
          f"MAE={fold_records[-1]['mae']:.4f}  "
          f"rows={fold_records[-1]['n_rows']}  "
          f"patients={fold_records[-1]['n_unique_patients']}")

# ── Fold stats ────────────────────────────────────────────────────────────────
fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(OUT_DIR / "fold_stats.csv", index=False)
print(f"\nFold summary (mean RMSE={fold_df['rmse'].mean():.4f}, "
      f"std={fold_df['rmse'].std():.4f}):")
print(fold_df[["fold", "n_rows", "n_unique_patients", "rmse", "mae"]].to_string(index=False))

# ── OOF predictions dataframe ─────────────────────────────────────────────────
oof_df = X_raw[["patient_id", "cohort", "gene", "rater_id"]].copy()
oof_df["y_true"]    = y.values
oof_df["y_pred"]    = oof_pred
oof_df["abs_error"] = np.abs(oof_df["y_true"] - oof_df["y_pred"])
oof_df["sq_error"]  = (oof_df["y_true"] - oof_df["y_pred"]) ** 2
oof_df.to_csv(OUT_DIR / "oof_predictions.csv")

# ── Group-level error aggregation ─────────────────────────────────────────────
def group_errors(df, col):
    g = df.groupby(col, dropna=False)
    return (
        g.agg(
            n_obs=("abs_error", "count"),
            rmse=("sq_error",  lambda x: float(np.sqrt(x.mean()))),
            mae= ("abs_error", "mean"),
            y_mean=("y_true",  "mean"),
        )
        .sort_values("rmse", ascending=False)
        .reset_index()
    )

rmse_patient = group_errors(oof_df, "patient_id")
rmse_cohort  = group_errors(oof_df, "cohort")
rmse_gene    = group_errors(oof_df, "gene")
rmse_rater   = group_errors(oof_df, "rater_id")

rmse_patient.to_csv(OUT_DIR / "rmse_by_patient.csv", index=False)
rmse_cohort .to_csv(OUT_DIR / "rmse_by_cohort.csv",  index=False)
rmse_gene   .to_csv(OUT_DIR / "rmse_by_gene.csv",    index=False)
rmse_rater  .to_csv(OUT_DIR / "rmse_by_rater.csv",   index=False)

print("\nRMSE by cohort:")
print(rmse_cohort.to_string(index=False))
print("\nRMSE by gene:")
print(rmse_gene.to_string(index=False))
print("\nTop-10 rater RMSE:")
print(rmse_rater.head(10).to_string(index=False))

# ── Top-20 absolute errors ────────────────────────────────────────────────────
top_errors = (
    oof_df
    .nlargest(20, "abs_error")
    [["patient_id", "cohort", "gene", "rater_id",
      "y_true", "y_pred", "abs_error", "sq_error"]]
    .round(3)
)
top_errors.to_csv(OUT_DIR / "top_errors.csv")
print("\nTop-20 largest absolute errors:")
print(top_errors.to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. Fold RMSE / MAE bar chart
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].bar(fold_df["fold"], fold_df["rmse"], color="#3b82d4")
axes[0].axhline(fold_df["rmse"].mean(), color="red", linestyle="--", label=f"mean={fold_df['rmse'].mean():.2f}")
axes[0].set_xlabel("Fold"); axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE per fold")
axes[0].legend()
axes[1].bar(fold_df["fold"], fold_df["mae"], color="#7c5cd8")
axes[1].axhline(fold_df["mae"].mean(), color="red", linestyle="--", label=f"mean={fold_df['mae'].mean():.2f}")
axes[1].set_xlabel("Fold"); axes[1].set_ylabel("MAE"); axes[1].set_title("MAE per fold")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_fold_rmse_mae.png", dpi=100)
plt.close()
print("Saved: plot_fold_rmse_mae.png")

# 2. RMSE by patient (histogram of per-patient RMSE)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(rmse_patient["rmse"].dropna(), bins=60, color="#3b82d4", edgecolor="white")
ax.set_xlabel("Per-patient RMSE"); ax.set_ylabel("Number of patients")
ax.set_title("Distribution of per-patient RMSE (Skrub HGBR)")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_rmse_by_patient.png", dpi=100)
plt.close()
print("Saved: plot_rmse_by_patient.png")

# 3. RMSE vs n_obs per patient
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(rmse_patient["n_obs"], rmse_patient["rmse"],
           alpha=0.35, s=12, color="#3b82d4")
ax.set_xlabel("Observations per patient"); ax.set_ylabel("RMSE")
ax.set_title("Patient RMSE vs number of observations")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_rmse_vs_nobs.png", dpi=100)
plt.close()
print("Saved: plot_rmse_vs_nobs.png")

# 4. OOF predicted vs actual
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(oof_df["y_true"], oof_df["y_pred"], alpha=0.15, s=4, color="#3b82d4")
mn, mx = oof_df["y_true"].min(), oof_df["y_true"].max()
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="perfect")
ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
ax.set_title("OOF: predicted vs actual (Skrub HGBR)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_oof_pred_vs_actual.png", dpi=100)
plt.close()
print("Saved: plot_oof_pred_vs_actual.png")

# ── Console summary ───────────────────────────────────────────────────────────
overall_rmse = float(np.sqrt(oof_df["sq_error"].mean()))
overall_mae  = float(oof_df["abs_error"].mean())

print("\n" + "=" * 65)
print("ERROR ANALYSIS SUMMARY  (Skrub TableVectorizer + HGBR)")
print("=" * 65)
print(f"Overall OOF RMSE : {overall_rmse:.4f}")
print(f"Overall OOF MAE  : {overall_mae:.4f}")
print(f"\nWorst 5 patients by RMSE:")
print(rmse_patient.head(5)[["patient_id","n_obs","rmse","mae"]].to_string(index=False))
print(f"\nBest  5 patients by RMSE:")
print(rmse_patient.tail(5)[["patient_id","n_obs","rmse","mae"]].to_string(index=False))
print(f"\nAll outputs saved to: {OUT_DIR}")
