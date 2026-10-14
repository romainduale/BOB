# Experiment 11 — seed bagging audit for frozen Exp08
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

MODEL_SEEDS = [1, 17, 42]
N_FOLDS = 5

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _find_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find Exp08 script")

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

print(f"Frozen Exp08 definition: {exp08_path.name}")

X = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy(float)

patient_ids = X["patient_id"].to_numpy()
unique_patients = pd.unique(patient_ids)
all_idx = np.arange(len(X))

NUM_THREADS = int(os.environ.get("NUM_THREADS", "4"))
STAGE = os.environ.get("EXP11_STAGE", "DEV").upper().strip()

if STAGE == "DEV":
    SPLIT_SEEDS = [112358]
elif STAGE == "CONFIRM":
    SPLIT_SEEDS = [2718281, 3141592]
else:
    raise ValueError("EXP11_STAGE must be DEV or CONFIRM")

print("\n" + "=" * 84)
print("EXP11 — SEED BAGGING OF FROZEN EXP08")
print("=" * 84)
print(f"Stage        : {STAGE}")
print(f"Rows         : {len(X)}")
print(f"Patients     : {len(unique_patients)}")
print(f"Split seeds  : {SPLIT_SEEDS}")
print(f"Model seeds  : {MODEL_SEEDS}")
print(f"Folds/seed   : {N_FOLDS}")
print("Architecture : unchanged Exp08")
print("NO tuning")
print("=" * 84)

def make_random_group_folds(seed):
    rng = np.random.default_rng(seed)
    pats = np.array(unique_patients, copy=True)
    rng.shuffle(pats)
    chunks = np.array_split(pats, N_FOLDS)
    folds = []
    for chunk in chunks:
        mask = np.isin(patient_ids, chunk)
        va = all_idx[mask]
        tr = all_idx[~mask]
        assert set(patient_ids[tr]).isdisjoint(set(patient_ids[va]))
        folds.append((tr, va))
    return folds

rows = []
repeat_rows = []
t_all = time.time()

for repeat_idx, split_seed in enumerate(SPLIT_SEEDS, start=1):
    print(
        f"\n--- {STAGE} repeat {repeat_idx}/{len(SPLIT_SEEDS)} "
        f"| split seed={split_seed} ---",
        flush=True,
    )
    folds = make_random_group_folds(split_seed)
    oof_single = np.full(len(y), np.nan)
    oof_bag = np.full(len(y), np.nan)

    for fold_idx, (tr, va) in enumerate(folds, start=1):
        t0 = time.time()
        preds = []

        for j, model_seed in enumerate(MODEL_SEEDS, start=1):
            print(
                f"  fold {fold_idx}/5 | model seed {model_seed} ({j}/{len(MODEL_SEEDS)})",
                flush=True,
            )
            model = ShrunkExpertRegressor(
                num_threads=NUM_THREADS,
                random_state=model_seed,
            )
            model.fit(X.iloc[tr], y[tr])
            preds.append(model.predict(X.iloc[va]))

        p_single = preds[0]
        p_bag = np.mean(np.vstack(preds), axis=0)

        oof_single[va] = p_single
        oof_bag[va] = p_bag

        r_single = rmse(y[va], p_single)
        r_bag = rmse(y[va], p_bag)
        delta = r_bag - r_single
        sec = time.time() - t0

        rows.append({
            "stage": STAGE,
            "repeat": repeat_idx,
            "split_seed": split_seed,
            "fold": fold_idx,
            "rmse_single": r_single,
            "rmse_bag": r_bag,
            "delta_bag_minus_single": delta,
            "bag_wins": delta < 0,
            "seconds": sec,
        })

        print(
            f"R{repeat_idx} F{fold_idx}: "
            f"single={r_single:.4f} | bag={r_bag:.4f} | "
            f"delta={delta:+.4f} | "
            f"{'BAG WINS' if delta < 0 else 'SINGLE WINS'} | "
            f"{sec:.0f}s",
            flush=True,
        )

    g_single = rmse(y, oof_single)
    g_bag = rmse(y, oof_bag)
    g_delta = g_bag - g_single

    repeat_rows.append({
        "stage": STAGE,
        "repeat": repeat_idx,
        "split_seed": split_seed,
        "global_rmse_single": g_single,
        "global_rmse_bag": g_bag,
        "global_delta_bag_minus_single": g_delta,
    })

    print(
        f"{STAGE} repeat {repeat_idx} GLOBAL: "
        f"single={g_single:.4f} | bag={g_bag:.4f} | delta={g_delta:+.4f}",
        flush=True,
    )

fold_df = pd.DataFrame(rows)
repeat_df = pd.DataFrame(repeat_rows)

suffix = STAGE.lower()
fold_path = RESULTS_DIR / f"11_seed_bagging_{suffix}_folds.csv"
repeat_path = RESULTS_DIR / f"11_seed_bagging_{suffix}_repeats.csv"
fold_df.to_csv(fold_path, index=False)
repeat_df.to_csv(repeat_path, index=False)

wins = int(fold_df["bag_wins"].sum())
nfolds = len(fold_df)
win_rate = wins / nfolds
mean_fold_delta = float(fold_df["delta_bag_minus_single"].mean())
std_fold_delta = float(fold_df["delta_bag_minus_single"].std(ddof=1))
mean_global_delta = float(repeat_df["global_delta_bag_minus_single"].mean())

if STAGE == "DEV":
    gd = float(repeat_df.iloc[0]["global_delta_bag_minus_single"])
    verdict = (
        "DEV_PASS__FREEZE_AND_RUN_CONFIRM"
        if (gd <= -0.008 and wins >= 3)
        else "DEV_FAIL__KEEP_SINGLE_EXP08"
    )
else:
    all_better = bool((repeat_df["global_delta_bag_minus_single"] < 0).all())
    verdict = (
        "CONFIRMED_BAGGING_GAIN"
        if (all_better and wins >= 7 and mean_global_delta <= -0.006)
        else "NOT_CONFIRMED__KEEP_SINGLE_EXP08"
    )

lines = [
    "",
    "=" * 84,
    f"EXP11 {STAGE} AUDIT",
    "=" * 84,
    f"Paired folds                     : {nfolds}",
    f"Bag wins                         : {wins}/{nfolds} ({100*win_rate:.1f}%)",
    f"Mean fold delta (bag - single)   : {mean_fold_delta:+.4f}",
    f"Std fold delta                   : {std_fold_delta:.4f}",
    "",
]

for _, rr in repeat_df.iterrows():
    lines.append(
        f"Repeat {int(rr['repeat'])} seed={int(rr['split_seed'])}: "
        f"single={rr['global_rmse_single']:.4f} | "
        f"bag={rr['global_rmse_bag']:.4f} | "
        f"delta={rr['global_delta_bag_minus_single']:+.4f}"
    )

lines += [
    "",
    f"Mean GLOBAL delta                : {mean_global_delta:+.4f}",
    f"VERDICT                          : {verdict}",
    "",
]

if STAGE == "DEV":
    lines += [
        "Pre-registered DEV gate:",
        "  global delta <= -0.008 AND >= 3/5 folds won.",
        "  If it passes, do not modify model seeds or Exp08 before CONFIRM.",
    ]
else:
    lines += [
        "Pre-registered CONFIRM rule:",
        "  both repeats globally better",
        "  >= 7/10 folds won",
        "  mean global delta <= -0.006",
    ]

lines += [
    "",
    f"Saved fold results   : {fold_path}",
    f"Saved repeat results : {repeat_path}",
    f"Total runtime        : {(time.time()-t_all)/60:.1f} min",
    "=" * 84,
]

summary = "\n".join(lines)
print(summary)

summary_path = RESULTS_DIR / f"11_seed_bagging_{suffix}_summary.txt"
summary_path.write_text(summary, encoding="utf-8")
print(f"Saved summary        : {summary_path}")
