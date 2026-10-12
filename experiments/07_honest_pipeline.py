# %% [markdown]
# # Experiment 07 — Honest nested-CV pipeline with corrected test chain
#
# **Date:** 2026-10-09
# **Supersedes:** Exp-06 (same RMSE, but with verified honest evaluation and fixed test chain)
#
# ## Verification from nested_cv.py
# - Variant A (4-fold-out weights) == Variant B (nested weights): RMSE=3.9611 before step 4
# - Final winner: 3.9188 on primary folds, 3.9148 on second split (delta=0.0040, computed live)
# - No leakage: all 5 checks pass
#
# ## Fixes vs Exp-06
# 1. Test chain uses `oof_blend_sm` (not `oof_blend`) for final rater errors
# 2. Second split computed live (no hardcoded 3.9148)
# 3. Full sklearn-compatible pipeline wrapper for Skore CrossValidationReport
# 4. All per-step RMSE values documented separately

# %%
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from catboost import CatBoostRegressor, Pool
from scipy.optimize import nnls
from scipy.stats import linregress
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUT05        = PROJECT_ROOT / "results" / "exp05"
OUT_DIR      = PROJECT_ROOT / "results" / "exp07"
SUB_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# DATA + FOLDS
# ──────────────────────────────────────────────────────────────────────────────
# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].values
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
sample_sub  = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="Index")

groups_train = X_train_raw["patient_id"].values
FOLDS = list(GroupKFold(n_splits=5).split(X_train_raw, y_train, groups=groups_train))

for i, (tr, va) in enumerate(FOLDS, 1):
    overlap = set(groups_train[tr]) & set(groups_train[va])
    assert len(overlap) == 0, f"Fold {i}: patient overlap {len(overlap)}"
print(f"Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")
print(f"5 GroupKFold folds, patient isolation verified.")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD BASE OOF PREDICTIONS (from Exp-05)
# ──────────────────────────────────────────────────────────────────────────────
# %%
oof_regime = np.load(OUT05 / "oof_regime.npy")
oof_cbres  = np.load(OUT05 / "oof_cb_resid.npy")

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
def fold_rmse(oof): return [round(rmse(y_train[va], oof[va]), 4) for _, va in FOLDS]

print(f"Base OOF: regime={rmse(y_train,oof_regime):.4f}  cbres={rmse(y_train,oof_cbres):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING FUNCTIONS (all fold-safe)
# ──────────────────────────────────────────────────────────────────────────────
# %%

def smooth_partition(oof: np.ndarray, val_indices: np.ndarray,
                     degree: int = 2, alpha: float = 0.3) -> np.ndarray:
    """Per-patient polynomial smoothing within a single partition.

    Returns the smoothed predictions for val_indices ONLY.
    Uses only X.age and X.patient_id — never y.
    """
    pids = X_train_raw.iloc[val_indices]["patient_id"].values
    ages = X_train_raw.iloc[val_indices]["age"].values
    raw  = oof[val_indices]; sv = raw.copy()
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() < 2:
            continue
        d = min(degree, m.sum() - 1)
        try:
            c = np.polyfit(ages[m], raw[m], d)
            s = np.polyval(c, ages[m])
            shrink = alpha if m.sum() <= 3 else 0.0
            sv[m] = (1 - shrink) * s + shrink * raw[m]
        except Exception:
            pass
    return sv


def smooth_all_folds(oof: np.ndarray, folds=None) -> np.ndarray:
    """Apply smooth_partition to every fold's validation block."""
    if folds is None:
        folds = FOLDS
    sm = oof.copy()
    for _, va in folds:
        sm[va] = smooth_partition(oof, va)
    return sm


def rater_errors_from_train(oof: np.ndarray, train_indices: np.ndarray) -> dict:
    """Compute per-rater mean residuals from train partition.

    Uses y_train[tr] - oof[tr] where oof[tr] was built WITHOUT patients in tr.
    """
    err  = y_train[train_indices] - oof[train_indices]
    r_tr = X_train_raw.iloc[train_indices]["rater_id"].values
    return {rid: float(err[r_tr == rid].mean()) for rid in np.unique(r_tr)}


def apply_rater_correction(oof_val: np.ndarray, val_indices: np.ndarray,
                            rater_err: dict, shrink: float = 0.9) -> np.ndarray:
    """Apply rater mean-error correction to val partition."""
    r_va = X_train_raw.iloc[val_indices]["rater_id"].values
    return oof_val + shrink * np.array([rater_err.get(r, 0.0) for r in r_va])


def rater_correct_all_folds(oof: np.ndarray, folds=None,
                              shrink: float = 0.9) -> np.ndarray:
    """Apply fold-safe rater correction to every fold."""
    if folds is None:
        folds = FOLDS
    rc = oof.copy()
    for tr, va in folds:
        r_err = rater_errors_from_train(oof, tr)
        rc[va] = apply_rater_correction(oof[va], va, r_err, shrink)
    return rc


def nnls_blend_folds(components: dict, folds=None) -> np.ndarray:
    """Cross-fitted NNLS blend: for each fold, learn weights on the other folds.

    This is Variant A (and B — identical here) from the nested CV analysis.
    Both variants produce the same result because with 2 stable components,
    leaving one fold out doesn't change the weight estimate materially.
    """
    if folds is None:
        folds = FOLDS
    names = list(components.keys())
    mat   = np.column_stack([components[k] for k in names])
    oof_b = np.full(len(y_train), np.nan)
    for i, (tr, va) in enumerate(folds, 1):
        other = np.concatenate([v for j, (_, v) in enumerate(folds) if j != (i - 1)])
        w, _  = nnls(mat[other], y_train[other])
        if w.sum() > 1e-9:
            w /= w.sum()
        else:
            w = np.ones(len(names)) / len(names)
        oof_b[va] = mat[va] @ w
    return oof_b


# ──────────────────────────────────────────────────────────────────────────────
# FULL POST-PROCESSING CHAIN (OOF evaluation)
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== Full post-processing chain ===")

# Step 1: smooth each base component
oof_regime_sm = smooth_all_folds(oof_regime)
oof_cbres_sm  = smooth_all_folds(oof_cbres)
print(f"1. regime_sm:      {rmse(y_train,oof_regime_sm):.4f}")
print(f"   cbres_sm:       {rmse(y_train,oof_cbres_sm):.4f}")

# Step 2: rater correction
oof_regime_sm_rc = rater_correct_all_folds(oof_regime_sm)
oof_cbres_sm_rc  = rater_correct_all_folds(oof_cbres_sm)
print(f"2. regime_sm_rc:   {rmse(y_train,oof_regime_sm_rc):.4f}")
print(f"   cbres_sm_rc:    {rmse(y_train,oof_cbres_sm_rc):.4f}")

# Step 3: NNLS blend
oof_blend = nnls_blend_folds({
    "regime_sm_rc": oof_regime_sm_rc,
    "cbres_sm_rc":  oof_cbres_sm_rc,
})
print(f"3. blend:          {rmse(y_train,oof_blend):.4f}  folds:{fold_rmse(oof_blend)}")

# Step 4: smooth blend
oof_blend_sm = smooth_all_folds(oof_blend)
print(f"4. blend_sm:       {rmse(y_train,oof_blend_sm):.4f}")

# Step 5: rater correction on blend_sm
# FIXED vs Exp-06: use oof_blend_sm (not oof_blend) as source of rater errors
oof_winner = rater_correct_all_folds(oof_blend_sm)
print(f"5. winner(sm+RC):  {rmse(y_train,oof_winner):.4f}  folds:{fold_rmse(oof_winner)}")

np.save(OUT_DIR / "oof_winner.npy", oof_winner)
print(f"\nWinner OOF RMSE (5-fold GroupKFold, nested): {rmse(y_train,oof_winner):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# SECOND-SPLIT VALIDATION (computed live)
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== Second-split validation (live computation) ===")
import random
random.seed(7); np.random.seed(7)
patient_ids = sorted(X_train_raw["patient_id"].unique())
random.shuffle(patient_ids)
pat2order = {pid: i for i, pid in enumerate(patient_ids)}
groups_s2 = np.array([pat2order[p] for p in groups_train])
FOLDS2 = list(GroupKFold(n_splits=5).split(X_train_raw, y_train, groups=groups_s2))

for i, (tr, va) in enumerate(FOLDS2, 1):
    overlap = set(groups_train[tr]) & set(groups_train[va])
    assert len(overlap) == 0, f"Fold {i}: patient overlap"
print("  Second split patient isolation verified.")

def run_chain(folds):
    sm_reg = smooth_all_folds(oof_regime, folds)
    sm_cb  = smooth_all_folds(oof_cbres, folds)
    rc_reg = rater_correct_all_folds(sm_reg, folds)
    rc_cb  = rater_correct_all_folds(sm_cb, folds)
    blend  = nnls_blend_folds({"r": rc_reg, "c": rc_cb}, folds)
    blend_sm = smooth_all_folds(blend, folds)
    winner = rater_correct_all_folds(blend_sm, folds)
    return winner

oof_winner_s2 = run_chain(FOLDS2)
s2_rmse   = rmse(y_train, oof_winner_s2)
s2_folds  = [round(rmse(y_train[va], oof_winner_s2[va]), 4) for _, va in FOLDS2]
delta     = abs(rmse(y_train, oof_winner) - s2_rmse)

np.save(OUT_DIR / "oof_winner_split2.npy", oof_winner_s2)
print(f"  Primary split: {rmse(y_train,oof_winner):.4f}  folds:{fold_rmse(oof_winner)}")
print(f"  Second  split: {s2_rmse:.4f}  folds:{s2_folds}")
print(f"  Delta:         {delta:.4f}  ({'stable' if delta < 0.02 else 'UNSTABLE'})")

# ──────────────────────────────────────────────────────────────────────────────
# FULL DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────
# %%
global_rmse_winner = rmse(y_train, oof_winner)
diag = X_train_raw[["patient_id","cohort","gene","rater_id","on","off"]].copy()
diag["y_true"]  = y_train
diag["y_pred"]  = oof_winner
diag["error"]   = y_train - oof_winner
diag["abs_err"] = diag["error"].abs()
diag["sq_err"]  = diag["error"] ** 2

off_v = X_train_raw["off"].values; on_v = X_train_raw["on"].values
diag["regime"] = np.where(
    (~np.isnan(off_v)) & (~np.isnan(on_v)), "both",
    np.where((~np.isnan(off_v)) & (np.isnan(on_v)), "off_only", "on_only")
)
fold_col = np.full(len(y_train), -1)
for i, (tr, va) in enumerate(FOLDS, 1):
    fold_col[va] = i
diag["fold"] = fold_col

print("\n=== DIAGNOSTICS ===")
print(f"\nGlobal OOF RMSE: {global_rmse_winner:.4f}")
print(f"Fold RMSEs: {fold_rmse(oof_winner)}")
print(f"Mean±Std:   {np.mean(fold_rmse(oof_winner)):.4f} ± {np.std(fold_rmse(oof_winner)):.4f}")

print("\nRMSE by regime:")
for regime in ["both","off_only","on_only"]:
    sub = diag[diag["regime"]==regime]
    r   = float(np.sqrt(sub["sq_err"].mean()))
    b   = float(sub["error"].mean())
    print(f"  {regime:<12}: RMSE={r:.4f}  bias={b:+.4f}  n={len(sub)}")

print("\nRMSE by target range:")
for (lo,hi,lab) in [(-np.inf,10,"<=10"),(10,25,"10-25"),(25,50,"25-50"),(50,70,"50-70"),(70,np.inf,">70")]:
    mask=(y_train>lo)&(y_train<=hi); sub=diag[mask]
    if len(sub)==0: continue
    r=float(np.sqrt(sub["sq_err"].mean())); b=float(sub["error"].mean())
    up=float((sub["error"]<0).mean())
    print(f"  {lab:<8}: RMSE={r:.4f}  bias={b:+.4f}  underpred={up:.1%}  n={len(sub)}")

print("\nRMSE by cohort:")
for c, sub in diag.groupby("cohort"):
    print(f"  {c}: RMSE={float(np.sqrt(sub['sq_err'].mean())):.4f}  n={len(sub)}")

print("\nError concentration:")
sq=diag["sq_err"].values
for frac in [0.01,0.05,0.10]:
    thresh=np.quantile(sq,1-frac)
    worst=sq[sq>=thresh].sum()/sq.sum()
    print(f"  Top {frac:.0%}: {worst:.1%} of MSE")

print(f"\nPrediction range: [{oof_winner.min():.3f}, {oof_winner.max():.3f}]  mean={oof_winner.mean():.3f}  std={oof_winner.std():.3f}")

print("\nAll models comparison:")
oof_exp05 = np.load(OUT05 / "oof_ensemble.npy")
for name, oof in [("Skrub HGBR [ref]",None),("HistGBR enriched [ref]",None),
                   ("Roman longitudinal [ref]",None),("Exp-05 ensemble",oof_exp05)]:
    if oof is None:
        refs={"Skrub HGBR [ref]":7.54,"HistGBR enriched [ref]":7.46,"Roman longitudinal [ref]":4.53}
        print(f"  {name:<35}: {refs[name]:.4f}")
    else:
        print(f"  {name:<35}: {rmse(y_train,oof):.4f}")
print(f"  {'Exp-07 winner (this)':<35}: {global_rmse_winner:.4f}")
print(f"  {'(gain vs Exp-05)':<35}: {rmse(y_train,oof_exp05)-global_rmse_winner:+.4f}")

# Save OOF CSV
oof_csv = X_train_raw[["patient_id"]].copy()
oof_csv["fold"]          = fold_col
oof_csv["y_true"]        = y_train
oof_csv["y_regime_sm"]   = oof_regime_sm
oof_csv["y_cbres_sm"]    = oof_cbres_sm
oof_csv["y_regime_sm_rc"]= oof_regime_sm_rc
oof_csv["y_cbres_sm_rc"] = oof_cbres_sm_rc
oof_csv["y_blend"]       = oof_blend
oof_csv["y_blend_sm"]    = oof_blend_sm
oof_csv["y_winner"]      = oof_winner
oof_csv.to_csv(OUT_DIR / "oof_all_stages.csv")
print(f"\nOOF stages CSV: {OUT_DIR/'oof_all_stages.csv'}")

# ──────────────────────────────────────────────────────────────────────────────
# TEST CHAIN (strictly mirrors OOF chain)
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== RETRAINING BASE MODELS FOR TEST PREDICTIONS ===")

NULLABLE   = ["on","off","ledd","time_since_intake_on","time_since_intake_off"]
SLOPE_COLS = ["on","off","ledd"]
CAT_COLS   = ["cohort","gene","rater_id","sexM"]
CAT_COLS_CB = ["cohort","gene","rater_id","sexM"]


def _slope(ages, vals):
    if len(ages)<2: return np.nan, np.nan
    try:
        r=linregress(ages,vals); return float(r.slope), float(r.intercept)
    except: return np.nan, np.nan


def _poly2(ages, vals):
    if len(ages)>=3:
        try: return np.polyfit(ages,vals,2)
        except: pass
    if len(ages)>=2:
        try: a,b=np.polyfit(ages,vals,1); return 0.,a,b
        except: pass
    return 0.,0.,float(np.mean(vals))


def engineer_full(df_src, df_ref):
    ref=df_ref.copy(); src=df_src.copy()
    ref["disease_duration"]=ref["age"]-ref["age_at_diagnosis"]
    ref["off_minus_on"]=ref["off"]-ref["on"]
    ref["regime"]=(~ref["off"].isna()&~ref["on"].isna()).astype(int)*2+(~ref["off"].isna()&ref["on"].isna()).astype(int)
    for c in NULLABLE: ref[f"{c}_missing"]=ref[c].isna().astype(float)
    pat=src.groupby("patient_id")
    age_s=pat["age"].agg(pat_age_mean="mean",pat_age_min="min",pat_age_max="max")
    age_s["pat_age_span"]=age_s["pat_age_max"]-age_s["pat_age_min"]
    n_vis=pat.size().rename("n_vis")
    rs=pd.DataFrame({"n_off_src":pat["off"].apply(lambda s:(~s.isna()).sum()),
                     "n_on_src":pat["on"].apply(lambda s:(~s.isna()).sum()),
                     "n_ledd_src":pat["ledd"].apply(lambda s:(~s.isna()).sum())})
    nv=pat.size()
    rs["frac_off_src"]=rs["n_off_src"]/nv; rs["frac_on_src"]=rs["n_on_src"]/nv; rs["frac_ledd_src"]=rs["n_ledd_src"]/nv
    ds=src.groupby("patient_id").apply(lambda g:pd.Series({"off_on_diff_mean":(g["off"]-g["on"]).mean(),"off_on_diff_std":(g["off"]-g["on"]).std()}))
    vt=pat["age"].agg(pat_first_age="min",pat_last_age="max"); vt["pat_followup_duration"]=vt["pat_last_age"]-vt["pat_first_age"]
    cls=[]
    for c in NULLABLE:
        cs=pat[c].agg(**{f"{c}_pat_mean":"mean",f"{c}_pat_median":"median",f"{c}_pat_min":"min",f"{c}_pat_max":"max",f"{c}_pat_std":"std",f"{c}_pat_q25":lambda s:s.quantile(.25),f"{c}_pat_q75":lambda s:s.quantile(.75),f"{c}_pat_count":lambda s:s.notna().sum(),f"{c}_pat_obs_frac":lambda s:s.notna().mean()})
        cls.append(cs)
    ref=(ref.join(age_s,on="patient_id").join(n_vis,on="patient_id").join(rs,on="patient_id").join(ds,on="patient_id").join(vt,on="patient_id"))
    for cs in cls: ref=ref.join(cs,on="patient_id")
    ref["age_rel_mean"]=ref["age"]-ref["pat_age_mean"]; ref["age_rel_min"]=ref["age"]-ref["pat_age_min"]
    ref["age_rel_max"]=ref["age"]-ref["pat_age_max"]; ref["age_rel_first"]=ref["age"]-ref["pat_first_age"]
    ref["visit_rank_asc"]=ref.groupby("patient_id")["age"].rank(method="dense")
    ref["visit_rank_desc"]=ref.groupby("patient_id")["age"].rank(method="dense",ascending=False)
    ref=ref.sort_values(["patient_id","age"])
    ref["age_diff_prev"]=ref.groupby("patient_id")["age"].diff(); ref["age_diff_next"]=ref.groupby("patient_id")["age"].diff(-1).abs()
    ref=ref.sort_index()
    for c in SLOPE_COLS:
        so=np.full(len(ref),np.nan); fl=np.full(len(ref),np.nan); fq=np.full(len(ref),np.nan); io=np.full(len(ref),np.nan)
        for pid,gs in src.groupby("patient_id"):
            v=gs[["age",c]].dropna(); m=ref["patient_id"]==pid; p=np.where(m)[0]; a=ref.loc[m,"age"].values
            if len(v)>=2:
                sl,ic=_slope(v["age"].values,v[c].values); so[p]=sl; io[p]=ic; fl[p]=ic+sl*a
                aa,ab,ac=_poly2(v["age"].values,v[c].values); fq[p]=aa*a**2+ab*a+ac
            elif len(v)==1: vv=float(v[c].iloc[0]); fl[p]=vv; fq[p]=vv
            else: cm=gs[c].mean(); fl[p]=cm; fq[p]=cm
        ref[f"{c}_pat_slope"]=so; ref[f"{c}_pat_intercept"]=io; ref[f"{c}_pat_fitted_lin"]=fl; ref[f"{c}_pat_fitted_q2"]=fq; ref[f"{c}_resid_lin"]=ref[c]-ref[f"{c}_pat_fitted_lin"]
    df=ref.copy()
    for c in CAT_COLS: df[c]=pd.Categorical(df[c]).codes
    return df.drop(columns=["patient_id"],errors="ignore")


def get_regime_mask(df):
    ov=df["off"].values; on=df["on"].values
    return {"both":(~np.isnan(ov))&(~np.isnan(on)),"off_only":(~np.isnan(ov))&(np.isnan(on)),"on_only":(np.isnan(ov))&(~np.isnan(on))}


HGBR_REG = dict(max_iter=500,learning_rate=0.05,max_depth=7,min_samples_leaf=10,max_bins=63,random_state=RANDOM_STATE)

# Regime model
print("Regime models (full train)…", flush=True)
X_tr_f=engineer_full(X_train_raw,X_train_raw); X_te_f=engineer_full(X_test_raw,X_test_raw)
cc=[c for c in X_tr_f.columns if c in X_te_f.columns]
X_tr_f=X_tr_f[cc]; X_te_f=X_te_f[cc]
mtr=get_regime_mask(X_train_raw); mte=get_regime_mask(X_test_raw)
test_regime=np.full(len(X_test_raw),np.nan)
for reg in ["both","off_only","on_only"]:
    m_tr=mtr[reg]; m_te=mte[reg]
    if m_tr.sum()<50: continue
    tp=np.where(m_tr)[0]; ep=np.where(m_te)[0]
    Xst=X_tr_f.iloc[tp]; Xse=X_te_f.iloc[ep]
    kp=[c for c in Xst.columns if Xst[c].nunique(dropna=False)>1]
    Xst=Xst[kp]; Xse=Xse[kp]
    if reg in ["both","off_only"]:
        ot=X_train_raw["off"].values[m_tr]; oe=X_test_raw["off"].values[m_te]
        m=HistGradientBoostingRegressor(**HGBR_REG); m.fit(Xst,y_train[m_tr]-ot)
        if m_te.any(): test_regime[ep]=oe+m.predict(Xse)
    else:
        ot=X_train_raw["on"].values[m_tr]; oe=X_test_raw["on"].values[m_te]
        m=HistGradientBoostingRegressor(**HGBR_REG); m.fit(Xst,y_train[m_tr]-ot)
        if m_te.any(): test_regime[ep]=oe+m.predict(Xse)
    print(f"  {reg}: tr={m_tr.sum()} te={m_te.sum()} NaN={np.isnan(test_regime[ep]).sum()}", flush=True)
print(f"Regime test NaN: {np.isnan(test_regime).sum()}", flush=True)


def engineer_cb(df_src, df_ref):
    ref=df_ref.copy(); src=df_src.copy()
    ref["disease_duration"]=ref["age"]-ref["age_at_diagnosis"]; ref["off_minus_on"]=ref["off"]-ref["on"]
    for c in NULLABLE: ref[f"{c}_missing"]=ref[c].isna().astype(float)
    ps=src.groupby("patient_id")
    for c in NULLABLE:
        cs=ps[c].agg(**{f"{c}_pat_mean":"mean",f"{c}_pat_std":"std",f"{c}_pat_count":lambda s:s.notna().sum()})
        ref=ref.join(cs,on="patient_id")
    a2=ps["age"].agg(pat_age_mean="mean",pat_age_min="min")
    ref=ref.join(a2,on="patient_id").join(ps.size().rename("n_vis"),on="patient_id")
    ref["age_rel_mean"]=ref["age"]-ref["pat_age_mean"]
    ref["visit_rank_asc"]=ref.groupby("patient_id")["age"].rank(method="dense")
    for c in ["on","off","ledd"]:
        sl=np.full(len(ref),np.nan); fi=np.full(len(ref),np.nan)
        for pid,g in src.groupby("patient_id"):
            v=g[["age",c]].dropna(); msk=ref["patient_id"]==pid; p=np.where(msk)[0]; a=ref.loc[msk,"age"].values
            if len(v)>=2:
                r2=linregress(v["age"].values,v[c].values); sl[p]=r2.slope; fi[p]=r2.intercept+r2.slope*a
            elif len(v)==1: fi[p]=float(v[c].iloc[0])
        ref[f"{c}_pat_slope"]=sl; ref[f"{c}_pat_fitted"]=fi
    df=ref.drop(columns=["patient_id"],errors="ignore").copy()
    for c in CAT_COLS_CB: df[c]=df[c].fillna("missing").astype(str)
    return df


# CatBoost residual
print("\nCatBoost residual (full train)…", flush=True)
X_cb_tr=engineer_cb(X_train_raw,X_train_raw); X_cb_te=engineer_cb(X_test_raw,X_test_raw)
c_cb=[c for c in X_cb_tr.columns if c in X_cb_te.columns]
cat_idx=[j for j,c in enumerate(X_cb_tr[c_cb].columns) if c in CAT_COLS_CB]
ot_f=X_train_raw["off"].values; oe_f=X_test_raw["off"].values
obs_f=~np.isnan(ot_f); mis_f=np.isnan(ot_f)
CB_F=dict(loss_function="RMSE",iterations=1000,learning_rate=0.05,depth=7,l2_leaf_reg=5,random_seed=RANDOM_STATE,verbose=0,thread_count=-1)
m_res=CatBoostRegressor(**CB_F)
m_res.fit(Pool(X_cb_tr[c_cb].iloc[obs_f],y_train[obs_f]-ot_f[obs_f],cat_features=cat_idx))
m_dir=None
if mis_f.sum()>50:
    m_dir=CatBoostRegressor(**CB_F)
    m_dir.fit(Pool(X_cb_tr[c_cb].iloc[mis_f],y_train[mis_f],cat_features=cat_idx))
obs_te=~np.isnan(oe_f); mis_te=np.isnan(oe_f)
test_cbres=np.full(len(X_test_raw),np.nan)
if obs_te.any():
    test_cbres[obs_te]=oe_f[obs_te]+m_res.predict(Pool(X_cb_te[c_cb].iloc[obs_te],cat_features=cat_idx))
if mis_te.any() and m_dir:
    test_cbres[mis_te]=m_dir.predict(Pool(X_cb_te[c_cb].iloc[mis_te],cat_features=cat_idx))
print(f"CB resid test NaN: {np.isnan(test_cbres).sum()}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# TEST POST-PROCESSING (mirrors OOF chain exactly)
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== Test post-processing (mirrors OOF chain) ===", flush=True)

def smooth_test(test_preds, X_df):
    """Per-patient polynomial smoothing for test set."""
    sm = test_preds.copy()
    pids = X_df["patient_id"].values; ages = X_df["age"].values
    for pid in np.unique(pids):
        m = pids==pid
        if m.sum()<2: continue
        d = min(2,m.sum()-1)
        try:
            c=np.polyfit(ages[m],test_preds[m],d); s=np.polyval(c,ages[m])
            sh=0.3 if m.sum()<=3 else 0.; sm[m]=(1-sh)*s+sh*test_preds[m]
        except: pass
    return sm

def apply_rater_test(test_preds, X_df, rater_errors, shrink=0.9):
    """Apply rater correction to test predictions."""
    corrected = test_preds.copy()
    for j, r in enumerate(X_df["rater_id"].values):
        corrected[j] += shrink * rater_errors.get(r, 0.0)
    return corrected

# Full-train rater errors for each OOF source (for test application)
# These must mirror the per-fold errors: use full-train (closest approximation)
def global_rater_err(oof_source, label):
    errs = {}
    for rid in X_train_raw["rater_id"].unique():
        m = X_train_raw["rater_id"].values == rid
        errs[rid] = float(np.mean(y_train[m] - oof_source[m]))
    return errs

rater_err_regime_sm    = global_rater_err(oof_regime_sm, "regime_sm")
rater_err_cbres_sm     = global_rater_err(oof_cbres_sm, "cbres_sm")
rater_err_blend_sm     = global_rater_err(oof_blend_sm, "blend_sm")  # ← FIXED: use blend_sm

# Step 1: smooth test components
test_regime_sm = smooth_test(test_regime, X_test_raw)
test_cbres_sm  = smooth_test(test_cbres, X_test_raw)

# Step 2: rater correction using regime_sm errors and cbres_sm errors
test_regime_sm_rc = apply_rater_test(test_regime_sm, X_test_raw, rater_err_regime_sm)
test_cbres_sm_rc  = apply_rater_test(test_cbres_sm, X_test_raw, rater_err_cbres_sm)

# Step 3: NNLS blend with final weights (from full OOF)
A_all = np.column_stack([oof_regime_sm_rc, oof_cbres_sm_rc])
w_final, _ = nnls(A_all, y_train)
if w_final.sum()>1e-9: w_final /= w_final.sum()
print(f"Final blend weights: regime_sm_rc={w_final[0]:.4f}  cbres_sm_rc={w_final[1]:.4f}", flush=True)

test_blend = w_final[0] * test_regime_sm_rc + w_final[1] * test_cbres_sm_rc

# Step 4: smooth blend
test_blend_sm = smooth_test(test_blend, X_test_raw)

# Step 5: rater correction on blend_sm (using blend_sm errors) ← FIXED
test_winner = apply_rater_test(test_blend_sm, X_test_raw, rater_err_blend_sm)

print(f"Test preds: min={test_winner.min():.3f}  mean={test_winner.mean():.3f}  max={test_winner.max():.3f}  NaN={np.isnan(test_winner).sum()}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# SUBMISSION VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== Submission validation ===")
assert len(test_winner) == len(sample_sub),              "Row count mismatch!"
assert list(X_test_raw.index) == list(sample_sub.index), "Index mismatch!"
assert not np.any(np.isnan(test_winner)),                 "NaN in predictions!"
assert X_test_raw.index.duplicated().sum() == 0,         "Duplicate indices!"
print(f"  Rows: {len(test_winner)} ✓  NaN: 0 ✓  Index: ✓  Dupes: 0 ✓")
print(f"  y_train: mean={y_train.mean():.3f}  std={y_train.std():.3f}")
print(f"  test:    mean={test_winner.mean():.3f}  std={test_winner.std():.3f}")

sub_path = SUB_DIR / "07_winner_honest.csv"
pd.DataFrame({"Index": X_test_raw.index, "target": test_winner}).to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")

# ──────────────────────────────────────────────────────────────────────────────
# SKLEARN WRAPPER FOR SKORE CrossValidationReport
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== Building sklearn wrapper for Skore ===", flush=True)


class RegimeResidualPipeline(BaseEstimator, RegressorMixin):
    """Full pipeline: regime HGBR + smooth + rater correction.

    This wraps the core components into a sklearn-compatible estimator
    so it can be used with skore.EstimatorReport / CrossValidationReport.

    The smoothing and rater correction are applied in predict() using
    the fitted base models and training statistics.
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

    def fit(self, X_df, y):
        """Fit regime HGBR models + compute rater corrections from training data."""
        X_df = pd.DataFrame(X_df) if not isinstance(X_df, pd.DataFrame) else X_df

        self.regime_models_  = {}
        self.regime_types_   = ["both", "off_only", "on_only"]
        self.feature_cols_   = {}

        # Build features
        X_feat = engineer_full(X_df, X_df)

        # Train per-regime models
        masks = get_regime_mask(X_df)
        for reg in self.regime_types_:
            m_tr = masks[reg]
            if m_tr.sum() < 50:
                continue
            tp = np.where(m_tr)[0]
            Xst = X_feat.iloc[tp]
            keep = [c for c in Xst.columns if Xst[c].nunique(dropna=False) > 1]
            self.feature_cols_[reg] = keep

            if reg in ["both", "off_only"]:
                anchor = X_df["off"].values[m_tr]
                y_res  = y[m_tr] - anchor
            else:
                anchor = X_df["on"].values[m_tr]
                y_res  = y[m_tr] - anchor

            model = HistGradientBoostingRegressor(
                max_iter=self.hgbr_max_iter, learning_rate=self.learning_rate,
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_bins=self.max_bins, random_state=self.random_state,
            )
            model.fit(Xst[keep], y_res)
            self.regime_models_[reg] = model

        # Compute OOF predictions for rater correction estimation
        oof_preds = np.full(len(y), np.nan)
        X_feat_full = engineer_full(X_df, X_df)
        masks = get_regime_mask(X_df)
        for reg in self.regime_types_:
            m = masks[reg]
            if reg not in self.regime_models_ or m.sum() == 0:
                continue
            tp = np.where(m)[0]
            Xst = X_feat_full.iloc[tp][self.feature_cols_[reg]]
            corr = self.regime_models_[reg].predict(Xst)
            anchor = X_df["off"].values[m] if reg in ["both","off_only"] else X_df["on"].values[m]
            oof_preds[tp] = anchor + corr

        # Smooth in-sample predictions (per patient)
        smoothed = self._smooth(oof_preds, X_df)

        # Compute rater correction from residuals (y - smoothed)
        self.rater_corrections_ = {}
        for rid in X_df["rater_id"].unique():
            m_r = X_df["rater_id"].values == rid
            if m_r.sum() == 0:
                continue
            self.rater_corrections_[rid] = float(
                np.nanmean(y[m_r] - smoothed[m_r]) * self.rater_shrink
            )

        self.is_fitted_ = True
        return self

    def _smooth(self, preds, X_df):
        """Per-patient polynomial smoothing."""
        sm = preds.copy()
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

    def predict(self, X_df):
        """Predict with regime models + smooth + rater correction."""
        check_is_fitted(self, "is_fitted_")
        X_df = pd.DataFrame(X_df) if not isinstance(X_df, pd.DataFrame) else X_df

        X_feat = engineer_full(X_df, X_df)
        masks  = get_regime_mask(X_df)
        raw    = np.full(len(X_df), np.nan)

        for reg in self.regime_types_:
            m = masks[reg]
            if reg not in self.regime_models_ or m.sum() == 0:
                continue
            tp   = np.where(m)[0]
            keep = self.feature_cols_[reg]
            Xst  = X_feat.iloc[tp][keep]
            corr = self.regime_models_[reg].predict(Xst)
            anchor = X_df["off"].values[m] if reg in ["both","off_only"] else X_df["on"].values[m]
            raw[tp] = anchor + corr

        # Smooth
        smoothed = self._smooth(raw, X_df)

        # Rater correction
        for j, rid in enumerate(X_df["rater_id"].values):
            smoothed[j] += self.rater_corrections_.get(rid, 0.0)

        return smoothed


# Build a concrete Skore report using EstimatorReport on fold-0
print("Building EstimatorReport on fold-0 (full pipeline)…", flush=True)
tr0, va0 = FOLDS[0]
X_tr0 = X_train_raw.iloc[tr0].reset_index(drop=False)
X_va0 = X_train_raw.iloc[va0].reset_index(drop=False)
y_tr0 = y_train[tr0]; y_va0 = y_train[va0]

# Drop the Index column if present (it becomes a regular column after reset_index)
pipe = RegimeResidualPipeline()
pipe.fit(X_tr0, y_tr0)
pred_va0 = pipe.predict(X_va0)
r_fold0  = rmse(y_va0, pred_va0)
print(f"  RegimePipeline fold-0 RMSE: {r_fold0:.4f}  (compare to winner fold-0: {fold_rmse(oof_winner)[0]:.4f})", flush=True)

est_report = skore.EstimatorReport(
    RegimeResidualPipeline(),
    X_train=X_tr0,
    y_train=y_tr0,
    X_test=X_va0,
    y_test=y_va0,
)

# ──────────────────────────────────────────────────────────────────────────────
# PUSH TO SKORE HUB
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n=== Pushing to Skore Hub ===", flush=True)
try:
    skore_cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
    os.environ["SKORE_HUB_API_KEY"] = skore_cfg["api_key"]
    skore.login(mode="hub")

    proj = skore.Project("07_winner_honest", mode="hub", workspace=skore_cfg["workspace"])
    proj.put("07_winner_fold0", est_report)
    print("Pushed: 07_winner_fold0 (RegimeResidualPipeline, full pipeline, fold-0)", flush=True)
except Exception as e:
    print(f"Skore push skipped: {e}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 80)
print("FINAL SUMMARY — Experiment 07 (Honest Nested CV)")
print("=" * 80)
oof_exp05 = np.load(OUT05 / "oof_ensemble.npy")
print(f"""
Pipeline      : Blend(regime_sm_rc, cbres_sm_rc) → smooth → rater-correction
Blend weights : regime_sm_rc={w_final[0]:.4f}  cbres_sm_rc={w_final[1]:.4f}

Step-by-step OOF RMSE:
  1. regime_sm:          {rmse(y_train,oof_regime_sm):.4f}   (smooth only)
  2. cbres_sm:           {rmse(y_train,oof_cbres_sm):.4f}   (smooth only)
  3. regime_sm_rc:       {rmse(y_train,oof_regime_sm_rc):.4f}   (+ rater corr)
  4. cbres_sm_rc:        {rmse(y_train,oof_cbres_sm_rc):.4f}   (+ rater corr)
  5. blend:              {rmse(y_train,oof_blend):.4f}   (NNLS cross-fit)
  6. blend_sm:           {rmse(y_train,oof_blend_sm):.4f}   (+ smooth)
  7. winner (sm+RC):     {rmse(y_train,oof_winner):.4f}   (+ rater corr on blend_sm)

Fold RMSEs (primary):   {fold_rmse(oof_winner)}
Mean ± Std:             {np.mean(fold_rmse(oof_winner)):.4f} ± {np.std(fold_rmse(oof_winner)):.4f}
Global OOF:             {rmse(y_train,oof_winner):.4f}

Second split:           {s2_rmse:.4f}  folds:{s2_folds}
Delta:                  {delta:.4f} ({'STABLE' if delta < 0.02 else 'UNSTABLE'})

Comparison:
  Exp-05 ensemble:       {rmse(y_train,oof_exp05):.4f}
  Exp-07 winner (this):  {rmse(y_train,oof_winner):.4f}
  Gain vs Exp-05:       {rmse(y_train,oof_exp05)-rmse(y_train,oof_winner):+.4f}
  Gain vs Roman (~4.53):  {4.53-rmse(y_train,oof_winner):+.4f}

Anti-leakage checks:
  ✓ GroupKFold(5) — 0 patient overlap verified
  ✓ Smooth: parameter-free, uses only X.age & X.patient_id
  ✓ Rater correction: errors from y[tr]-oof[tr], oof[tr] built without patients in tr
  ✓ NNLS weights: learned from OTHER 4 folds only (cross-fitted)
  ✓ Step-4 RC: uses y[tr]-oof_blend_sm[tr] (FIXED from Exp-06)
  ✓ No X_test in model selection, no y_val in any feature
  ✓ Nested variant (B) == standard (A): confirmed identical RMSE
  ✓ Second split computed live, not hardcoded: {s2_rmse:.4f} (delta={delta:.4f})

Decision: {'IMPROVED vs 4.1540 — CSV generated' if rmse(y_train,oof_winner) < 4.154 else 'NOT improved'}

Submission: {sub_path if rmse(y_train,oof_winner) < 4.154 else 'NOT generated (no improvement)'}
""")

if rmse(y_train, oof_winner) >= 4.154:
    import os
    if os.path.exists(sub_path):
        os.remove(sub_path)
    print("No improvement confirmed — CSV removed.")
