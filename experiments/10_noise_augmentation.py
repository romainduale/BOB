# Experiment 10 — train-only measurement-noise augmentation audit
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _find_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any candidate Exp08 file")


def load_definitions_only(path, stop_markers):
    source = path.read_text(encoding="utf-8")
    cut = len(source)
    for marker in stop_markers:
        pos = source.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    if cut == len(source):
        raise RuntimeError(f"Could not locate DATA section in {path.name}")
    ns = {"__file__": str(path), "__name__": f"_defs_{path.stem}"}
    exec(compile(source[:cut], str(path), "exec"), ns)
    return ns


exp08_path = _find_existing([
    EXPERIMENTS_DIR / "08_shrunk_expert_skore_FINAL.py",
    EXPERIMENTS_DIR / "08_shrunk_expert.py",
])
ns08 = load_definitions_only(
    exp08_path,
    ("# DATA + PATIENT", "# DATA + PATIENT-LEVEL", "# DATA + PATIENT-GROUPED"),
)
ShrunkExpertRegressor = ns08["ShrunkExpertRegressor"]
rater_cal = ns08["rater_cal"]
print(f"Frozen Exp08 definition: {exp08_path.name}")

SEVERITY_BINS = np.array([-np.inf, 20.0, 30.0, 40.0, 50.0, 60.0, np.inf])
NOISE_SCALE = 0.25


def _raw_rater_residuals(df, y, channel):
    cal = rater_cal(df, y)
    table, glob = cal[channel]
    a = df["rater_id"].map(table["a"]).fillna(glob[1]).to_numpy(float)
    b = df["rater_id"].map(table["b"]).fillna(glob[0]).to_numpy(float)
    observed = df[channel].notna().to_numpy()
    mu = a + b * y
    resid = df[channel].to_numpy(float) - mu
    return observed, resid


def _build_residual_pools(df, y):
    pools = {}
    severity_bin = np.digitize(y, SEVERITY_BINS[1:-1], right=False)
    for channel in ("on", "off"):
        observed, resid = _raw_rater_residuals(df, y, channel)
        for b in range(len(SEVERITY_BINS) - 1):
            z = resid[observed & (severity_bin == b)]
            z = z[np.isfinite(z)]
            if len(z) >= 100:
                pools[(channel, b)] = z
        zg = resid[observed]
        pools[(channel, "global")] = zg[np.isfinite(zg)]
    return pools, severity_bin


def augment_training_fold(X, y, seed, noise_scale=NOISE_SCALE):
    X0 = X.copy().reset_index(drop=True)
    y0 = np.asarray(y, dtype=float).copy()
    pools, severity_bin = _build_residual_pools(X0, y0)
    rng = np.random.default_rng(seed)
    Xa = X0.copy()
    Xa["patient_id"] = Xa["patient_id"].astype(str) + f"__aug_{seed}"

    for channel in ("on", "off"):
        vals = Xa[channel].to_numpy(float).copy()
        obs = np.isfinite(vals)
        jitter = np.zeros(len(Xa), dtype=float)
        for b in range(len(SEVERITY_BINS) - 1):
            idx = np.flatnonzero(obs & (severity_bin == b))
            if len(idx) == 0:
                continue
            pool = pools.get((channel, b), pools[(channel, "global")])
            jitter[idx] = noise_scale * rng.choice(pool, size=len(idx), replace=True)
        vals[obs] += jitter[obs]
        vals[obs] = np.clip(vals[obs], 0.0, None)
        Xa[channel] = vals

    X_aug = pd.concat([X0, Xa], ignore_index=True)
    y_aug = np.concatenate([y0, y0])
    return X_aug, y_aug


class NoiseAugmentedExp08:
    def __init__(self, num_threads=4, random_state=1, aug_seed=12345):
        self.num_threads = num_threads
        self.random_state = random_state
        self.aug_seed = aug_seed

    def fit(self, X, y):
        X_aug, y_aug = augment_training_fold(X, y, self.aug_seed)
        self.model_ = ShrunkExpertRegressor(
            num_threads=self.num_threads,
            random_state=self.random_state,
        )
        self.model_.fit(X_aug, y_aug)
        return self

    def predict(self, X):
        return self.model_.predict(X)


X = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy(float)
patient_ids = X["patient_id"].to_numpy()
unique_patients = pd.unique(patient_ids)
all_idx = np.arange(len(X))

NUM_THREADS = int(os.environ.get("NUM_THREADS", "4"))
STAGE = os.environ.get("EXP10_STAGE", "DEV").upper().strip()
if STAGE == "DEV":
    SEEDS = [424242]
elif STAGE == "CONFIRM":
    SEEDS = [8675309, 1618033]
else:
    raise ValueError("EXP10_STAGE must be DEV or CONFIRM")

N_FOLDS = 5
print("\n" + "=" * 82)
print("EXP10 — TRAIN-ONLY MEASUREMENT-NOISE AUGMENTATION")
print("=" * 82)
print(f"Stage            : {STAGE}")
print(f"Rows             : {len(X)}")
print(f"Patients         : {len(unique_patients)}")
print(f"Seeds            : {SEEDS}")
print(f"Noise scale      : {NOISE_SCALE}")
print("Augmented copies : 1")
print("Missingness      : unchanged")
print("Validation       : untouched original patients")
print("NO tuning")
print("=" * 82)


def make_random_group_folds(seed):
    rng = np.random.default_rng(seed)
    pats = np.array(unique_patients, copy=True)
    rng.shuffle(pats)
    chunks = np.array_split(pats, N_FOLDS)
    folds = []
    for chunk in chunks:
        valid_mask = np.isin(patient_ids, chunk)
        va = all_idx[valid_mask]
        tr = all_idx[~valid_mask]
        assert set(patient_ids[tr]).isdisjoint(set(patient_ids[va]))
        folds.append((tr, va))
    return folds


rows = []
repeat_rows = []
start_all = time.time()

for repeat_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n--- {STAGE} repeat {repeat_idx}/{len(SEEDS)} | split seed={seed} ---", flush=True)
    folds = make_random_group_folds(seed)
    oof08 = np.full(len(y), np.nan)
    oof10 = np.full(len(y), np.nan)

    for fold_idx, (tr, va) in enumerate(folds, start=1):
        t0 = time.time()
        m08 = ShrunkExpertRegressor(num_threads=NUM_THREADS, random_state=1)
        m08.fit(X.iloc[tr], y[tr])
        p08 = m08.predict(X.iloc[va])

        m10 = NoiseAugmentedExp08(
            num_threads=NUM_THREADS,
            random_state=1,
            aug_seed=int(seed * 100 + fold_idx),
        )
        m10.fit(X.iloc[tr], y[tr])
        p10 = m10.predict(X.iloc[va])

        oof08[va] = p08
        oof10[va] = p10
        r08 = rmse(y[va], p08)
        r10 = rmse(y[va], p10)
        delta = r10 - r08
        sec = time.time() - t0
        rows.append({
            "stage": STAGE,
            "repeat": repeat_idx,
            "seed": seed,
            "fold": fold_idx,
            "rmse_08": r08,
            "rmse_10": r10,
            "delta_10_minus_08": delta,
            "exp10_wins": delta < 0,
            "seconds": sec,
        })
        print(
            f"R{repeat_idx} F{fold_idx}: 08={r08:.4f} | 10={r10:.4f} | "
            f"delta={delta:+.4f} | {'10 WINS' if delta < 0 else '08 WINS'} | {sec:.0f}s",
            flush=True,
        )

    global08 = rmse(y, oof08)
    global10 = rmse(y, oof10)
    global_delta = global10 - global08
    repeat_rows.append({
        "stage": STAGE,
        "repeat": repeat_idx,
        "seed": seed,
        "global_rmse_08": global08,
        "global_rmse_10": global10,
        "global_delta_10_minus_08": global_delta,
    })
    print(
        f"{STAGE} repeat {repeat_idx} GLOBAL: 08={global08:.4f} | 10={global10:.4f} | "
        f"delta={global_delta:+.4f}", flush=True
    )

fold_df = pd.DataFrame(rows)
repeat_df = pd.DataFrame(repeat_rows)
suffix = STAGE.lower()
fold_path = RESULTS_DIR / f"10_noise_aug_{suffix}_folds.csv"
repeat_path = RESULTS_DIR / f"10_noise_aug_{suffix}_repeats.csv"
fold_df.to_csv(fold_path, index=False)
repeat_df.to_csv(repeat_path, index=False)

wins = int(fold_df["exp10_wins"].sum())
nfolds = len(fold_df)
win_rate = wins / nfolds
mean_fold_delta = float(fold_df["delta_10_minus_08"].mean())
std_fold_delta = float(fold_df["delta_10_minus_08"].std(ddof=1)) if nfolds > 1 else float("nan")
mean_global_delta = float(repeat_df["global_delta_10_minus_08"].mean())

if STAGE == "DEV":
    dev_delta = float(repeat_df.iloc[0]["global_delta_10_minus_08"])
    verdict = "DEV_PASS__FREEZE_EXP10_AND_RUN_CONFIRM" if dev_delta <= -0.015 else "DEV_FAIL__STOP_EXP10"
else:
    all_repeats_better = bool((repeat_df["global_delta_10_minus_08"] < 0).all())
    verdict = "CONFIRMED_ROBUST_GAIN" if (all_repeats_better and wins >= 8 and mean_global_delta <= -0.020) else "NOT_CONFIRMED__KEEP_EXP08"

lines = [
    "",
    "=" * 82,
    f"EXP10 {STAGE} AUDIT",
    "=" * 82,
    f"Paired folds                 : {nfolds}",
    f"Exp10 wins                   : {wins}/{nfolds} ({100*win_rate:.1f}%)",
    f"Mean fold delta (10 - 08)    : {mean_fold_delta:+.4f}",
    f"Std fold delta               : {std_fold_delta:.4f}",
    "",
]
for _, rr in repeat_df.iterrows():
    lines.append(
        f"Repeat {int(rr['repeat'])} seed={int(rr['seed'])}: "
        f"08={rr['global_rmse_08']:.4f} | 10={rr['global_rmse_10']:.4f} | "
        f"delta={rr['global_delta_10_minus_08']:+.4f}"
    )
lines += [
    "",
    f"Mean GLOBAL delta            : {mean_global_delta:+.4f}",
    f"VERDICT                      : {verdict}",
    "",
]
if STAGE == "DEV":
    lines += [
        "Pre-registered DEV gate:",
        "  Continue only if global delta <= -0.015.",
        "  If it passes, DO NOT modify Exp10 before CONFIRM.",
    ]
else:
    lines += [
        "Pre-registered CONFIRM rule:",
        "  - both repeats globally better",
        "  - >= 8/10 paired folds won",
        "  - mean global delta <= -0.020",
    ]
lines += [
    "",
    f"Saved fold results   : {fold_path}",
    f"Saved repeat results : {repeat_path}",
    f"Total runtime        : {(time.time()-start_all)/60:.1f} min",
    "=" * 82,
]
summary = "\n".join(lines)
print(summary)
summary_path = RESULTS_DIR / f"10_noise_aug_{suffix}_summary.txt"
summary_path.write_text(summary, encoding="utf-8")
print(f"Saved summary        : {summary_path}")
