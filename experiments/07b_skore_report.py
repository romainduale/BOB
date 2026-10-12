# %% [markdown]
# # Experiment 07b — Skore report for the full Exp-07 pipeline
#
# Uses skore.evaluate() with GroupKFold splitter to get a proper cross-validated
# report of the RegimeResidualPipeline on the 5 patient-grouped folds.

# %%
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from scipy.stats import linregress
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "results" / "exp07"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────
# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values
groups      = X_train_raw["patient_id"].values

FOLDS = list(GroupKFold(n_splits=5).split(X_train_raw, y_train, groups=groups))
def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (shared helpers — same as Exp-07)
# ──────────────────────────────────────────────────────────────────────────────
# %%
NULLABLE   = ["on","off","ledd","time_since_intake_on","time_since_intake_off"]
SLOPE_COLS = ["on","off","ledd"]
CAT_COLS   = ["cohort","gene","rater_id","sexM"]


def _slope(ages, vals):
    if len(ages) < 2: return np.nan, np.nan
    try:
        r = linregress(ages, vals); return float(r.slope), float(r.intercept)
    except: return np.nan, np.nan


def engineer_full(df_src, df_ref):
    ref = df_ref.copy(); src = df_src.copy()
    ref["disease_duration"] = ref["age"] - ref["age_at_diagnosis"]
    ref["off_minus_on"]     = ref["off"] - ref["on"]
    ref["regime"] = (
        (~ref["off"].isna() & ~ref["on"].isna()).astype(int)*2 +
        (~ref["off"].isna() &  ref["on"].isna()).astype(int)
    )
    for c in NULLABLE:
        ref[f"{c}_missing"] = ref[c].isna().astype(float)
    pat = src.groupby("patient_id")
    age_s = pat["age"].agg(pat_age_mean="mean", pat_age_min="min", pat_age_max="max")
    age_s["pat_age_span"] = age_s["pat_age_max"] - age_s["pat_age_min"]
    n_vis = pat.size().rename("n_vis")
    for c in NULLABLE:
        cs = pat[c].agg(**{f"{c}_pat_mean":"mean", f"{c}_pat_std":"std",
                           f"{c}_pat_count": lambda s: s.notna().sum()})
        ref = ref.join(cs, on="patient_id")
    ref = ref.join(age_s, on="patient_id").join(n_vis, on="patient_id")
    ref["age_rel_mean"]  = ref["age"] - ref["pat_age_mean"]
    ref["visit_rank_asc"] = ref.groupby("patient_id")["age"].rank(method="dense")
    for c in SLOPE_COLS:
        sl = np.full(len(ref), np.nan); fi = np.full(len(ref), np.nan)
        for pid, g in src.groupby("patient_id"):
            v = g[["age",c]].dropna(); msk = ref["patient_id"]==pid
            pos = np.where(msk)[0]; ag = ref.loc[msk,"age"].values
            if len(v)>=2:
                sl2, ic = _slope(v["age"].values, v[c].values)
                sl[pos] = sl2; fi[pos] = ic + sl2*ag
            elif len(v)==1:
                fi[pos] = float(v[c].iloc[0])
        ref[f"{c}_pat_slope"] = sl; ref[f"{c}_pat_fitted"] = fi
    df = ref.copy()
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c]).codes
    return df.drop(columns=["patient_id","Index"], errors="ignore")


def get_regime_mask(df):
    ov = df["off"].values; on = df["on"].values
    return {
        "both":     (~np.isnan(ov)) & (~np.isnan(on)),
        "off_only": (~np.isnan(ov)) & ( np.isnan(on)),
        "on_only":  ( np.isnan(ov)) & (~np.isnan(on)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SKLEARN ESTIMATOR WRAPPING THE FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
# %%

class RegimeResidualPipeline(BaseEstimator, RegressorMixin):
    """Full pipeline: regime HGBR + smooth + rater correction.

    sklearn-compatible estimator for use with skore.evaluate / skore.EstimatorReport.

    Parameters
    ----------
    hgbr_max_iter : int
    learning_rate : float
    max_depth : int
    min_samples_leaf : int
    max_bins : int
    smooth_degree : int
    smooth_alpha : float
        Shrinkage toward raw prediction for patients with ≤3 visits.
    rater_shrink : float
        Regularization on rater mean-error correction.
    random_state : int
    """

    def __init__(self, hgbr_max_iter=500, learning_rate=0.05, max_depth=7,
                 min_samples_leaf=10, max_bins=63, smooth_degree=2,
                 smooth_alpha=0.3, rater_shrink=0.9, random_state=42):
        self.hgbr_max_iter    = hgbr_max_iter
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins         = max_bins
        self.smooth_degree    = smooth_degree
        self.smooth_alpha     = smooth_alpha
        self.rater_shrink     = rater_shrink
        self.random_state     = random_state

    def fit(self, X, y):
        X_df = self._to_df(X)
        n    = len(X_df)

        self.regime_models_  = {}
        self.feature_cols_   = {}
        self.regime_types_   = ["both", "off_only", "on_only"]

        X_feat = engineer_full(X_df, X_df)
        masks  = get_regime_mask(X_df)

        for reg in self.regime_types_:
            m_tr = masks[reg]
            if m_tr.sum() < 50:
                continue
            tp   = np.where(m_tr)[0]
            Xst  = X_feat.iloc[tp]
            keep = [c for c in Xst.columns if Xst[c].nunique(dropna=False) > 1]
            self.feature_cols_[reg] = keep

            if reg in ["both", "off_only"]:
                anchor = X_df["off"].values[m_tr]
            else:
                anchor = X_df["on"].values[m_tr]

            model = HistGradientBoostingRegressor(
                max_iter=self.hgbr_max_iter, learning_rate=self.learning_rate,
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_bins=self.max_bins, random_state=self.random_state,
            )
            model.fit(Xst[keep], y[m_tr] - anchor)
            self.regime_models_[reg] = model

        # Compute in-sample predictions for rater error estimation
        raw = np.full(n, np.nan)
        for reg in self.regime_types_:
            m = masks[reg]
            if reg not in self.regime_models_ or m.sum() == 0:
                continue
            tp    = np.where(m)[0]
            Xst   = X_feat.iloc[tp][self.feature_cols_[reg]]
            corr  = self.regime_models_[reg].predict(Xst)
            anchor = X_df["off"].values[m] if reg in ["both","off_only"] else X_df["on"].values[m]
            raw[tp] = anchor + corr

        smoothed = self._smooth(raw, X_df)

        self.rater_corrections_ = {}
        for rid in X_df["rater_id"].unique():
            m_r = X_df["rater_id"].values == rid
            if m_r.sum() == 0:
                continue
            self.rater_corrections_[rid] = (
                float(np.nanmean(y[m_r] - smoothed[m_r])) * self.rater_shrink
            )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X_df  = self._to_df(X)
        X_feat = engineer_full(X_df, X_df)
        masks  = get_regime_mask(X_df)
        raw    = np.full(len(X_df), np.nan)

        for reg in self.regime_types_:
            m = masks[reg]
            if reg not in self.regime_models_ or m.sum() == 0:
                continue
            tp    = np.where(m)[0]
            keep  = self.feature_cols_[reg]
            Xst   = X_feat.iloc[tp][keep]
            corr  = self.regime_models_[reg].predict(Xst)
            anchor = X_df["off"].values[m] if reg in ["both","off_only"] else X_df["on"].values[m]
            raw[tp] = anchor + corr

        smoothed = self._smooth(raw, X_df)

        for j, rid in enumerate(X_df["rater_id"].values):
            smoothed[j] += self.rater_corrections_.get(rid, 0.0)

        return smoothed

    def _smooth(self, preds, X_df):
        sm   = preds.copy()
        pids = X_df["patient_id"].values if "patient_id" in X_df.columns else np.zeros(len(preds))
        ages = X_df["age"].values
        for pid in np.unique(pids):
            m = pids == pid
            if m.sum() < 2:
                continue
            d = min(self.smooth_degree, m.sum() - 1)
            try:
                c = np.polyfit(ages[m], preds[m], d)
                s = np.polyval(c, ages[m])
                sh = self.smooth_alpha if m.sum() <= 3 else 0.0
                sm[m] = (1 - sh) * s + sh * preds[m]
            except Exception:
                pass
        return sm

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE WITH SKORE (skore.evaluate = CrossValidationReport)
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("Evaluating RegimeResidualPipeline with skore.evaluate …", flush=True)

# Use the same GroupKFold folds as OOF evaluation
gkf    = GroupKFold(n_splits=5)
folds  = list(gkf.split(X_train_raw, y_train, groups=groups))

# Quick manual fold-by-fold check before pushing
print("Manual fold-by-fold RMSE:", flush=True)
fold_rmses = []
for i, (tr, va) in enumerate(folds, 1):
    pipe = RegimeResidualPipeline()
    pipe.fit(X_train_raw.iloc[tr], y_train[tr])
    preds = pipe.predict(X_train_raw.iloc[va])
    r = rmse(y_train[va], preds)
    fold_rmses.append(r)
    print(f"  Fold {i}: {r:.4f}", flush=True)

print(f"  Mean: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}", flush=True)
print(f"  Global (weighted): {rmse(y_train, np.concatenate([pipe.predict(X_train_raw.iloc[va]) for _,va in folds])):.4f}", flush=True)

# Note: the pipeline RMSE (~4.0) is slightly above the full Exp-07 winner (3.9188)
# because the winner also uses CatBoost residual (cbres_sm_rc, weight ~0.22).
# The pipeline wraps only the REGIME component.
# This is honest: the regime component alone gives ~4.08 without the blend.

# ──────────────────────────────────────────────────────────────────────────────
# PUSH TO SKORE HUB
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\nPushing to Skore Hub …", flush=True)
try:
    skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
    os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]
    skore.login(mode="hub")

    # Use skore.evaluate with the full grouped CV splitter
    report = skore.evaluate(
        RegimeResidualPipeline(),
        X_train_raw,
        y_train,
        splitter=iter(folds),
        n_jobs=1,
    )

    proj = skore.Project("07_winner_honest", mode="hub", workspace=skore_cfg["workspace"])
    proj.put("07_regime_pipeline_cv5", report)
    print("Pushed: 07_regime_pipeline_cv5 (5-fold GroupKFold CV of full pipeline)", flush=True)

    # Per-fold RMSE from the report
    fold_rmses_skore = [float(r.metrics.rmse()) for r in report.reports_]
    print(f"Skore fold RMSE: {[round(r,4) for r in fold_rmses_skore]}", flush=True)
    print(f"Skore mean RMSE: {np.mean(fold_rmses_skore):.4f}", flush=True)

except Exception as e:
    print(f"Skore push failed: {e}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
# %%
oof_exp05   = np.load(OUT_DIR.parent / "exp05" / "oof_ensemble.npy")
oof_winner  = np.load(OUT_DIR / "oof_winner.npy")

print(f"""
=== SUMMARY Exp-07 Skore report ===

RegimeResidualPipeline (regime HGBR + smooth + rater correction)
  Fold RMSEs (manual):  {[round(r,4) for r in fold_rmses]}
  Mean ± Std:           {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}

  Note: Pipeline wraps the REGIME component only (~80% weight in full winner).
        Full winner (regime + CatBoost blend) OOF: {rmse(y_train, oof_winner):.4f}
        Exp-05 ensemble:                           {rmse(y_train, oof_exp05):.4f}
        Gain (full winner vs Exp-05):             {rmse(y_train,oof_exp05)-rmse(y_train,oof_winner):+.4f}

  Skore report pushed to Hub: 07_regime_pipeline_cv5
  Submission: submissions/07_winner_honest.csv
""")
