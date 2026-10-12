# %% [markdown]
# # Experiment 05 — Final blend, sklearn wrapper + Skore Hub report
#
# Purpose:
#   - wrap the final calibrated longitudinal ensemble in a sklearn-compatible estimator;
#   - evaluate it on a patient-disjoint GroupKFold split with skore.EstimatorReport;
#   - optionally rerun honest 5-fold patient-level CV with RUN_FULL_CV=1;
#   - upload the held-out report to Skore Hub under project "05_final_blend".
#
# IMPORTANT: all target-dependent rater calibration is learned inside fit(X, y).
# predict(X) uses only the fitted calibration + the covariates/trajectory of X.

import json
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import skore
from skore import Project, login
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.hub import load_skore_credentials

warnings.filterwarnings("ignore")

def rater_cal(df,yv,tf=lambda v:v):
    """per-rater linear forward model  s = a + b*target  for on and off (fit on given rows)"""
    out={}
    for c in ["on","off"]:
        m=df[c].notna().values
        rows=[]
        yv_=tf(yv); sv=tf(df[c].values)
        glob=np.polyfit(yv_[m],sv[m],1)
        for r,idx in df[m].groupby("rater_id").indices.items():
            yy=yv_[m][idx]; ss=sv[m][idx]
            if len(idx)>=30: b,a=np.polyfit(yy,ss,1)
            else: b,a=glob
            rows.append((r,a,b))
        t=pd.DataFrame(rows,columns=["rater_id","a","b"]).set_index("rater_id")
        out[c]=(t,glob)
    return out
def apply_cal(df,cal,tf=lambda v:v):
    r={}
    for c in ["on","off"]:
        t,glob=cal[c]
        a=df.rater_id.map(t.a).fillna(glob[1]).values; b=df.rater_id.map(t.b).fillna(glob[0]).values
        r[c]=(tf(df[c].values)-a)/b
    return r
def _fit(age,s,deg,lam=0.0):
    m=~np.isnan(s); n=m.sum()
    if n==0: return np.full(len(age),np.nan),np.nan
    if n<=deg or n<3: return np.full(len(age),s[m].mean()),np.nan
    a0=age[m].mean(); x=age[m]-a0
    V=np.vander(x,deg+1)
    P=np.eye(deg+1)*0; 
    if deg==2: P[0,0]=lam
    c=np.linalg.solve(V.T@V+P+1e-9*np.eye(deg+1),V.T@s[m])
    return np.polyval(c,age-a0), c[-2]
def sig_feats(age,s,pref,F,sl):
    m=~np.isnan(s); n=int(m.sum())
    F[pref+"_cnt"][sl]=n
    if n>0:
        v=s[m]; F[pref+"_mean"][sl]=v.mean(); F[pref+"_med"][sl]=np.median(v)
        F[pref+"_max"][sl]=v.max(); F[pref+"_min"][sl]=v.min(); F[pref+"_std"][sl]=v.std() if n>1 else np.nan
        F[pref+"_last"][sl]=v[-1]; F[pref+"_first"][sl]=v[0]
    f1,sl1=_fit(age,s,1); F[pref+"_lin"][sl]=f1; F[pref+"_slope"][sl]=sl1
    f2,_=_fit(age,s,2,lam=20.0*0+ 30.0); F[pref+"_quad"][sl]=f2
    for h in (1.0,2.5):
        if n>0:
            W=np.exp(-0.5*((age[:,None]-age[m][None,:])/h)**2)
            ws=W.sum(1); F[f"{pref}_k{h}"][sl]=(W@s[m])/ws; F[f"{pref}_kw{h}"][sl]=ws
def build(df,cal,calq=None):
    cs=apply_cal(df,cal)
    n=len(df); pid=df.patient_id.values
    bnd=np.flatnonzero(np.r_[True,pid[1:]!=pid[:-1],True])
    age=df.age.values
    s_on,s_off=cs["on"],cs["off"]
    w_on,w_off=0.4,1.0
    num=np.nan_to_num(s_on)*w_on*~np.isnan(s_on)+np.nan_to_num(s_off)*w_off*~np.isnan(s_off)
    den=w_on*~np.isnan(s_on)+w_off*~np.isnan(s_off)
    s_comb=np.where(den>0,num/np.where(den>0,den,1),np.nan)
    sigs={}
    if calq is not None:
        tf=np.sqrt; cq=apply_cal(df,calq,tf); q_on,q_off=cq["on"],cq["off"]
        # precision weights: residual sd in sqrt space (on noisier)
        wq_on,wq_off=0.45,1.0
        nq=np.nan_to_num(q_on)*wq_on*~np.isnan(q_on)+np.nan_to_num(q_off)*wq_off*~np.isnan(q_off)
        dq=wq_on*~np.isnan(q_on)+wq_off*~np.isnan(q_off)
        q_cb=np.where(dq>0,nq/np.where(dq>0,dq,1),np.nan)
        sigs.update({"qon":q_on,"qoff":q_off,"qcb":q_cb})
    sigs.update({"on":s_on,"off":s_off,"cb":s_comb,"raw_off":df.off.values,"raw_on":df.on.values,"ledd":df.ledd.values})
    names=[]
    for p in sigs:
        for k in ["cnt","mean","med","max","min","std","last","first","lin","slope","quad","k1.0","kw1.0","k2.5","kw2.5"]:
            names.append(f"{p}_{k}")
    F={k:np.full(n,np.nan) for k in names}
    for p in ["on","off","ton","toff"]:
        for k in ["miss_k1.5","miss_frac"]: F[f"{p}_{k}"]=np.full(n,np.nan)
    F["visit_idx"]=np.zeros(n); F["n_visits"]=np.zeros(n); F["span"]=np.zeros(n); F["age_rel_min"]=np.zeros(n); F["age_rel_max"]=np.zeros(n)
    tsi_on=df.time_since_intake_on.values; tsi_off=df.time_since_intake_off.values
    tmiss={"on":np.isnan(df.on.values),"off":np.isnan(df.off.values),"ton":np.isnan(tsi_on),"toff":np.isnan(tsi_off)}
    for a,b in zip(bnd[:-1],bnd[1:]):
        sl=slice(a,b); ag=age[sl]
        for p,v in sigs.items(): sig_feats(ag,v[sl],p,F,sl)
        W=np.exp(-0.5*((ag[:,None]-ag[None,:])/1.5)**2); W/=W.sum(1,keepdims=True)
        for p,mm in tmiss.items():
            x=mm[sl].astype(float); F[f"{p}_miss_k1.5"][sl]=W@x; F[f"{p}_miss_frac"][sl]=x.mean()
        F["visit_idx"][sl]=np.arange(b-a); F["n_visits"][sl]=b-a; F["span"][sl]=ag[-1]-ag[0]
        F["age_rel_min"][sl]=ag-ag[0]; F["age_rel_max"][sl]=ag-ag[-1]
    X=pd.DataFrame(F,index=df.index)
    X["age"]=df.age.values; X["age_dx"]=df.age_at_diagnosis.values; X["dur"]=X.age-X.age_dx
    X["cohort"]=(df.cohort=="B").astype(int).values; X["sexM"]=df.sexM.values
    X["gene"]=pd.Categorical(df.gene,categories=["No Mutation","LRRK2+","GBA+","OTHER+"]).codes
    for c in ["on","off","ledd","time_since_intake_on","time_since_intake_off"]: X[c]=df[c].values
    X["cal_on"]=s_on; X["cal_off"]=s_off; X["cal_cb"]=s_comb
    X["off_minus_on"]=df.off.values-df.on.values
    X["rater_n"]=df.groupby("patient_id").rater_id.transform("nunique").values
    return X

OFF_L=[0,5,15,25,35,45,55,65,80,120]; OFF_S=[2.0,2.7,5.6,8.1,9.5,11.0,11.7,11.0,11.9,13.0]
ON_L=[0,10,15,25,35,45,55,65,80,120]; ON_S=[5.0,6.0,6.7,9.4,11.8,14.1,15.8,16.5,18.8,20.0]
def wfit(age,s,w,deg,lam):
    m=~np.isnan(s); n=m.sum()
    if n==0: return np.full(len(age),np.nan),np.nan
    if n<=deg or n<3: return np.full(len(age),np.average(s[m],weights=w[m])),np.nan
    ww=w[m]/w[m].mean(); a0=age[m].mean(); x=age[m]-a0; V=np.vander(x,deg+1)
    P=np.zeros((deg+1,deg+1)); 
    if deg==2: P[0,0]=lam
    c=np.linalg.solve(V.T@(V*ww[:,None])+P+1e-9*np.eye(deg+1),V.T@(ww*s[m]))
    return np.polyval(c,age-a0),c[-2]
def build_extra(df,cal,mode="hetero",w_on=0.2):
    cs=apply_cal(df,cal); so,sf=cs["on"],cs["off"]; n=len(df)
    pid=df.patient_id.values; age=df.age.values
    bnd=np.flatnonzero(np.r_[True,pid[1:]!=pid[:-1],True])
    mo,mf=~np.isnan(so),~np.isnan(sf)
    names=["lin","slope","quad","k1.0","k2.5","kp1.0","kp2.5","mean","prec"]
    F={f"hc_{k}":np.full(n,np.nan) for k in names}
    for a,b in zip(bnd[:-1],bnd[1:]):
        sl=slice(a,b); ag=age[sl]; o,f=so[sl],sf[sl]; om,fm=mo[sl],mf[sl]
        if mode=="hetero":
            # level estimate from provisional (w_on=.4) signal, kernel h=2.5
            wp=0.4*om+1.0*fm; sp=np.where(wp>0,(np.nan_to_num(o)*0.4*om+np.nan_to_num(f)*fm)/np.where(wp>0,wp,1),np.nan)
            mm=~np.isnan(sp)
            if mm.sum()>0:
                K=np.exp(-0.5*((ag[:,None]-ag[mm][None,:])/2.5)**2); lvl=(K@sp[mm])/K.sum(1)
            else: lvl=np.full(len(ag),37.5)
            lvl=np.clip(lvl,3,90)
            won=om/np.interp(lvl,ON_L,ON_S)**2; wof=fm/np.interp(lvl,OFF_L,OFF_S)**2
        else:
            won=om*w_on; wof=fm*1.0
        W=won+wof; s=np.where(W>0,(np.nan_to_num(o)*won+np.nan_to_num(f)*wof)/np.where(W>0,W,1),np.nan)
        wv=np.where(W>0,W,1e-12)
        F["hc_lin"][sl],F["hc_slope"][sl]=wfit(ag,s,wv,1,0); F["hc_quad"][sl],_=wfit(ag,s,wv,2,30.0)
        mk=W>0
        for h in (1.0,2.5):
            if mk.sum()>0:
                K=np.exp(-0.5*((ag[:,None]-ag[mk][None,:])/h)**2)*W[mk][None,:]
                F[f"hc_k{h}"][sl]=(K@s[mk])/K.sum(1); F[f"hc_kp{h}"][sl]=K.sum(1)
        if mk.sum()>0: F["hc_mean"][sl]=np.average(s[mk],weights=W[mk]); F["hc_prec"][sl]=W.sum()
    return pd.DataFrame(F,index=df.index)




def sort_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows patient-by-patient (first appearance), then by age; preserve Index."""
    o = pd.factorize(df["patient_id"])[0]
    return (
        df.assign(_patient_order=o)
        .sort_values(["_patient_order", "age"], kind="stable")
        .drop(columns="_patient_order")
    )


def smooth_traj(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Quadratic per-patient smoothing, then monotone projection in age."""
    pid = df["patient_id"].to_numpy()
    age = df["age"].to_numpy()
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    out = np.empty(len(pred), dtype=float)

    for a, b in zip(bnd[:-1], bnd[1:]):
        q = _fit(age[a:b], pred[a:b], 2, 0.0)[0]
        out[a:b] = IsotonicRegression(increasing=True).fit_transform(age[a:b], q)

    return out


class FinalBlendRegressor(RegressorMixin, BaseEstimator):
    """Final Bobathon model, fully sklearn-compatible.

    fit(X, y):
        learns rater calibrations and the four base learners from X/y only.

    predict(X):
        builds patient-level longitudinal features from X only, predicts with the
        fitted learners, smooths each learner's trajectory per patient, blends,
        clips at zero, and restores the original row order.
    """

    def __init__(
        self,
        lgb_a_rounds=1200,
        lgb_b_rounds=800,
        lgb_a_weight=0.413,
        lgb_b_weight=0.383,
        hgb_weight=0.089,
        hgb_hc_weight=0.116,
        num_threads=1,
        random_state=1,
    ):
        self.lgb_a_rounds = lgb_a_rounds
        self.lgb_b_rounds = lgb_b_rounds
        self.lgb_a_weight = lgb_a_weight
        self.lgb_b_weight = lgb_b_weight
        self.hgb_weight = hgb_weight
        self.hgb_hc_weight = hgb_hc_weight
        self.num_threads = num_threads
        self.random_state = random_state

    def _check_X(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FinalBlendRegressor requires X as a pandas DataFrame.")
        required = {
            "patient_id", "cohort", "sexM", "gene", "age_at_diagnosis", "age",
            "ledd", "time_since_intake_on", "time_since_intake_off",
            "rater_id", "on", "off",
        }
        missing = required.difference(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def fit(self, X, y):
        self._check_X(X)
        Xs = sort_visits(X.copy())
        ys = pd.Series(np.asarray(y), index=X.index).loc[Xs.index].to_numpy(dtype=float)

        # Target-dependent calibration is fitted ONLY here, on the estimator's train set.
        self.cal_ = rater_cal(Xs, ys)
        self.calq_ = rater_cal(Xs, ys, tf=np.sqrt)

        F = build(Xs, self.cal_)
        Fq = build(Xs, self.cal_, self.calq_)
        E = build_extra(Xs, self.cal_, "hetero")
        Fhc = pd.concat([F, E], axis=1)

        self.feature_columns_ = list(F.columns)
        self.sqrt_feature_columns_ = list(Fq.columns)
        self.hc_feature_columns_ = list(Fhc.columns)

        lgb_a = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=31,
            min_data_in_leaf=40,
            feature_fraction=0.6,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            verbose=-1,
            num_threads=self.num_threads,
            seed=self.random_state,
        )
        lgb_b = {
            **lgb_a,
            "learning_rate": 0.06,
            "feature_fraction": 0.3,
        }

        self.lgb_a_ = lgb.train(lgb_a, lgb.Dataset(F, ys), num_boost_round=self.lgb_a_rounds)
        self.lgb_b_ = lgb.train(lgb_b, lgb.Dataset(Fq, ys), num_boost_round=self.lgb_b_rounds)

        hgb_params = dict(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=0,
        )
        gene_idx = F.columns.get_loc("gene")
        self.hgb_ = HistGradientBoostingRegressor(
            categorical_features=[gene_idx], **hgb_params
        ).fit(F, ys)
        self.hgb_hc_ = HistGradientBoostingRegressor(**hgb_params).fit(Fhc, ys)

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        self._check_X(X)
        if not hasattr(self, "cal_"):
            raise RuntimeError("Estimator is not fitted.")

        original_index = X.index
        Xs = sort_visits(X.copy())

        F = build(Xs, self.cal_).reindex(columns=self.feature_columns_)
        Fq = build(Xs, self.cal_, self.calq_).reindex(columns=self.sqrt_feature_columns_)
        E = build_extra(Xs, self.cal_, "hetero")
        Fhc = pd.concat([F, E], axis=1).reindex(columns=self.hc_feature_columns_)

        pred = {
            "lgb": self.lgb_a_.predict(F),
            "lgbB_sqrt": self.lgb_b_.predict(Fq),
            "hgb": self.hgb_.predict(F),
            "hgb_hc": self.hgb_hc_.predict(Fhc),
        }

        weights = {
            "lgb": self.lgb_a_weight,
            "lgbB_sqrt": self.lgb_b_weight,
            "hgb": self.hgb_weight,
            "hgb_hc": self.hgb_hc_weight,
        }
        wsum = sum(weights.values())
        blended_sorted = sum(
            weights[k] * smooth_traj(Xs, pred[k]) for k in weights
        ) / wsum
        blended_sorted = np.clip(blended_sorted, 0.0, None)

        # restore the exact original X order expected by Skore / caller
        return pd.Series(blended_sorted, index=Xs.index).loc[original_index].to_numpy()


# ──────────────────────────────────────────────────────────────────────────────
# DATA + PATIENT-LEVEL SPLITS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
X_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy()
groups = X_raw["patient_id"].to_numpy()

folds = list(GroupKFold(n_splits=5).split(X_raw, y, groups=groups))
report_fold = int(os.environ.get("REPORT_FOLD", "0"))
if report_fold not in range(5):
    raise ValueError("REPORT_FOLD must be 0, 1, 2, 3, or 4")

# Optional exact 5-fold CV with the same wrapper.
# PowerShell: $env:RUN_FULL_CV="1"; python scripts/05_longitudinal_blend_skore.py
if os.environ.get("RUN_FULL_CV", "0") == "1":
    print("Running full 5-fold patient-level CV ...", flush=True)
    oof = np.full(len(y), np.nan)
    fold_rmses = []
    for i, (tr, va) in enumerate(folds, 1):
        est = FinalBlendRegressor()
        est.fit(X_raw.iloc[tr], y[tr])
        p = est.predict(X_raw.iloc[va])
        oof[va] = p
        r = float(np.sqrt(mean_squared_error(y[va], p)))
        fold_rmses.append(r)
        print(f"Fold {i}/5 RMSE = {r:.4f}", flush=True)
    global_rmse = float(np.sqrt(mean_squared_error(y, oof)))
    print(f"Fold RMSEs : {[round(v, 4) for v in fold_rmses]}")
    print(f"Global OOF RMSE : {global_rmse:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# SKORE ESTIMATOR REPORT ON ONE HONEST GROUPKFOLD HOLDOUT
# ──────────────────────────────────────────────────────────────────────────────
tr, va = folds[report_fold]
print(
    f"Building Skore report on fold {report_fold}: "
    f"train={len(tr)}, test={len(va)}, "
    f"train patients={X_raw.iloc[tr].patient_id.nunique()}, "
    f"test patients={X_raw.iloc[va].patient_id.nunique()}",
    flush=True,
)
assert set(X_raw.iloc[tr].patient_id).isdisjoint(set(X_raw.iloc[va].patient_id))

report = skore.EstimatorReport(
    FinalBlendRegressor(),
    X_train=X_raw.iloc[tr],
    y_train=y[tr],
    X_test=X_raw.iloc[va],
    y_test=y[va],
)

# ──────────────────────────────────────────────────────────────────────────────
# PUSH TO SKORE HUB
# ──────────────────────────────────────────────────────────────────────────────

load_skore_credentials()

cfg = json.loads((PROJECT_ROOT / ".skore").read_text())

login(mode="hub")

project = Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

report_key = "05_final_blend"

project.put(report_key, report)

print(f"Uploaded report key: {report_key}")
print("Skore Hub upload successful.")