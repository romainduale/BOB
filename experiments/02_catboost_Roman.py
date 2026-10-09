# %% [markdown]
# # Experiment 02 — CatBoost direct and residual
#
# **Date:** 2026-10-08
# **Goal:** Beat HistGBR CV RMSE 7.4530 with CatBoost's native categorical
# and NaN handling.  Two models: direct target prediction, and a residual
# model (`target - off` when off is observed; direct fallback otherwise).
#
# No Skore upload.  No hyperparameter search.  No blend.
# thread_count=-1 so CatBoost uses all available cores.

# %%
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Data + feature engineering (identical to Exp 01)

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

groups = X_train_raw["patient_id"].values

NUMERIC_COLS = [
    "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off",
]
MISSING_INDICATOR_COLS = ["off", "on", "ledd", "time_since_intake_on", "time_since_intake_off"]
CAT_COLS   = ["cohort", "gene", "rater_id", "sexM"]
NUMERIC_ENG = NUMERIC_COLS + ["disease_duration", "off_minus_on"]
INDICATOR_COLS = [f"{c}_missing" for c in MISSING_INDICATOR_COLS]
ALL_COLS = NUMERIC_ENG + CAT_COLS + INDICATOR_COLS


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Identical feature engineering to Exp 01."""
    df = df.copy().drop(columns=["patient_id"], errors="ignore")
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"]     = df["off"] - df["on"]
    for col in MISSING_INDICATOR_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(float)
    return df


def to_cb(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns and stringify categoricals (NaN → 'missing')."""
    df = df[ALL_COLS].copy()
    for c in CAT_COLS:
        df[c] = df[c].fillna("missing").astype(str)
    return df


X_eng   = engineer(X_train_raw)
X_te    = engineer(X_test_raw)
X_cb    = to_cb(X_eng)
X_cb_te = to_cb(X_te)

# raw off values kept for the residual model routing
off_raw_train = X_eng["off"].values
off_raw_test  = X_te["off"].values

# %% [markdown]
# ## GroupKFold — same 5 splits as Exp 01

# %%
FOLDS = list(GroupKFold(n_splits=5).split(X_eng, y_train, groups=groups))
print(f"Folds: {len(FOLDS)}, val sizes: {[len(v) for _, v in FOLDS]}")

# %% [markdown]
# ## Helpers

# %%
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def summary(fold_rmses: list) -> dict:
    return {
        "fold_rmses": [round(r, 4) for r in fold_rmses],
        "mean":       round(float(np.mean(fold_rmses)), 4),
        "std":        round(float(np.std(fold_rmses)),  4),
    }


CB = dict(
    loss_function="RMSE",
    iterations=1000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=False,
    thread_count=-1,   # use all available cores
)

# %% [markdown]
# ## Model A — CatBoost direct

# %%
print("=" * 60)
print("Model A: CatBoost direct")
print("=" * 60)
t0 = time.time()

oof_a = np.full(len(y_train), np.nan)
rmses_a = []

for i, (tr, va) in enumerate(FOLDS, 1):
    model = CatBoostRegressor(**CB)
    model.fit(Pool(X_cb.iloc[tr], y_train[tr], cat_features=CAT_COLS))
    preds = model.predict(Pool(X_cb.iloc[va], cat_features=CAT_COLS))
    oof_a[va] = preds
    fold_r = rmse(y_train[va], preds)
    rmses_a.append(fold_r)
    print(f"  fold {i}/5  RMSE={fold_r:.4f}", flush=True)

res_a = summary(rmses_a)
print(f"CatBoost direct  mean={res_a['mean']:.4f}  std={res_a['std']:.4f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Model B — CatBoost residual
#
# Training fold rows where `off` is observed → fit on `target - off`.
# Training fold rows where `off` is missing  → fit separate direct model.
# Validation predictions assembled per-row using the same routing.

# %%
print("=" * 60)
print("Model B: CatBoost residual")
print("=" * 60)
t0 = time.time()

oof_b = np.full(len(y_train), np.nan)
rmses_b = []

for i, (tr, va) in enumerate(FOLDS, 1):
    off_tr = off_raw_train[tr]
    obs_tr = ~np.isnan(off_tr)   # off observed in training fold
    mis_tr =  np.isnan(off_tr)   # off missing  in training fold

    # --- residual branch (off observed) ---
    model_res = CatBoostRegressor(**CB)
    model_res.fit(Pool(
        X_cb.iloc[tr][obs_tr],
        y_train[tr][obs_tr] - off_tr[obs_tr],
        cat_features=CAT_COLS,
    ))

    # --- direct fallback branch (off missing) ---
    model_dir = None
    if mis_tr.sum() > 0:
        model_dir = CatBoostRegressor(**CB)
        model_dir.fit(Pool(
            X_cb.iloc[tr][mis_tr],
            y_train[tr][mis_tr],
            cat_features=CAT_COLS,
        ))

    # --- assemble val predictions ---
    off_va = off_raw_train[va]
    obs_va = ~np.isnan(off_va)
    mis_va =  np.isnan(off_va)

    preds = np.full(len(va), np.nan)
    if obs_va.any():
        res_hat = model_res.predict(Pool(X_cb.iloc[va][obs_va], cat_features=CAT_COLS))
        preds[obs_va] = off_va[obs_va] + res_hat
    if mis_va.any():
        if model_dir is not None:
            preds[mis_va] = model_dir.predict(Pool(X_cb.iloc[va][mis_va], cat_features=CAT_COLS))
        else:
            preds[mis_va] = y_train[tr].mean()

    oof_b[va] = preds
    fold_r = rmse(y_train[va], preds)
    rmses_b.append(fold_r)
    print(f"  fold {i}/5  RMSE={fold_r:.4f}", flush=True)

res_b = summary(rmses_b)
print(f"CatBoost residual  mean={res_b['mean']:.4f}  std={res_b['std']:.4f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## Comparison

# %%
HIST_RMSE = 7.4530   # Exp-01 HistGBR mean CV RMSE

comparison = pd.DataFrame([
    {"Model": "HistGBR (Exp-01)",   "mean": HIST_RMSE, "std": None, "fold_rmses": "—"},
    {"Model": "CatBoost direct",    **res_a},
    {"Model": "CatBoost residual",  **res_b},
]).sort_values("mean").reset_index(drop=True)

print("\nCV RMSE comparison (GroupKFold 5, sorted by mean):")
print(comparison[["Model", "mean", "std", "fold_rmses"]].to_string(index=False))

# %% [markdown]
# ## Save OOF and test predictions for the best CatBoost model

# %%
scratch_dir = PROJECT_ROOT / "scratch" / "exp02"
scratch_dir.mkdir(parents=True, exist_ok=True)

oof_df = pd.DataFrame({
    "oof_cb_direct":   oof_a,
    "oof_cb_residual": oof_b,
    "y_true":          y_train,
}, index=X_train_raw.index)
oof_df.to_csv(scratch_dir / "oof_predictions.csv")
print(f"\nOOF saved → {scratch_dir / 'oof_predictions.csv'}")

# pick best by mean CV RMSE among the two CatBoost models
best_name, best_res = (
    ("CatBoost direct",   res_a) if res_a["mean"] <= res_b["mean"]
    else ("CatBoost residual", res_b)
)
print(f"Best CatBoost model: {best_name}  mean={best_res['mean']:.4f}")

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

if best_name == "CatBoost direct":
    final = CatBoostRegressor(**CB)
    final.fit(Pool(X_cb, y_train, cat_features=CAT_COLS))
    test_preds = final.predict(Pool(X_cb_te, cat_features=CAT_COLS))

else:   # residual
    off_full = off_raw_train
    obs_full = ~np.isnan(off_full)
    mis_full =  np.isnan(off_full)

    final_res = CatBoostRegressor(**CB)
    final_res.fit(Pool(X_cb.iloc[obs_full], y_train[obs_full] - off_full[obs_full], cat_features=CAT_COLS))

    final_dir = None
    if mis_full.sum() > 0:
        final_dir = CatBoostRegressor(**CB)
        final_dir.fit(Pool(X_cb.iloc[mis_full], y_train[mis_full], cat_features=CAT_COLS))

    obs_te = ~np.isnan(off_raw_test)
    mis_te =  np.isnan(off_raw_test)
    test_preds = np.full(len(off_raw_test), np.nan)
    if obs_te.any():
        test_preds[obs_te] = off_raw_test[obs_te] + final_res.predict(
            Pool(X_cb_te.iloc[obs_te], cat_features=CAT_COLS)
        )
    if mis_te.any() and final_dir is not None:
        test_preds[mis_te] = final_dir.predict(
            Pool(X_cb_te.iloc[mis_te], cat_features=CAT_COLS)
        )

submission = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds})
submission.to_csv(submissions_dir / "02_catboost.csv", index=False)
print(f"Submission → {submissions_dir / '02_catboost.csv'} ({len(submission)} rows)")
