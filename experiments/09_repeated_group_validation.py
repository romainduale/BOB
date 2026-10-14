# Experiment 09 — repeated randomized patient-group validation
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

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _find_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of these files:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )

def load_definitions_only(path, stop_markers):
    source = path.read_text(encoding="utf-8")
    cut = len(source)
    for marker in stop_markers:
        pos = source.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    if cut == len(source):
        raise RuntimeError(
            f"Could not locate DATA section in {path.name}; refusing to execute full script."
        )
    ns = {"__file__": str(path), "__name__": f"_defs_{path.stem}"}
    exec(compile(source[:cut], str(path), "exec"), ns)
    return ns

exp05_path = _find_existing([
    EXPERIMENTS_DIR / "05_longitudinal_blend_skore_FINAL.py",
    EXPERIMENTS_DIR / "05_longitudinal_blend_skore.py",
])
exp08_path = _find_existing([
    EXPERIMENTS_DIR / "08_shrunk_expert_skore_FINAL.py",
    EXPERIMENTS_DIR / "08_shrunk_expert.py",
])

stop_markers = (
    "# DATA + PATIENT",
    "# DATA + PATIENT-LEVEL",
    "# DATA + PATIENT-GROUPED",
)

ns05 = load_definitions_only(exp05_path, stop_markers)
ns08 = load_definitions_only(exp08_path, stop_markers)
FinalBlendRegressor = ns05["FinalBlendRegressor"]
ShrunkExpertRegressor = ns08["ShrunkExpertRegressor"]

print(f"Exp05 definition: {exp05_path.name}")
print(f"Exp08 definition: {exp08_path.name}")

X = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy(dtype=float)

patient_ids = X["patient_id"].to_numpy()
unique_patients = pd.unique(patient_ids)

N_REPEATS = int(os.environ.get("N_REPEATS", "3"))
if N_REPEATS < 1 or N_REPEATS > 10:
    raise ValueError("N_REPEATS must be between 1 and 10.")

SEEDS = [1729, 2718, 3141, 5772, 7919, 104729, 130363, 155921, 196613, 262147][:N_REPEATS]
N_FOLDS = 5
NUM_THREADS = int(os.environ.get("NUM_THREADS", "4"))

print()
print("=" * 78)
print("EXP09 — REPEATED RANDOMIZED PATIENT-GROUP VALIDATION")
print("=" * 78)
print(f"Rows         : {len(X)}")
print(f"Patients     : {len(unique_patients)}")
print(f"Repeats      : {N_REPEATS}")
print(f"Folds/repeat : {N_FOLDS}")
print(f"Paired folds : {N_REPEATS * N_FOLDS}")
print("NO tuning. Exp05 and Exp08 are frozen.")
print("=" * 78)

def make_random_group_folds(seed):
    rng = np.random.default_rng(seed)
    pats = np.array(unique_patients, copy=True)
    rng.shuffle(pats)
    chunks = np.array_split(pats, N_FOLDS)
    all_idx = np.arange(len(X))
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
grand_start = time.time()

for repeat_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n--- Repeat {repeat_idx}/{N_REPEATS} | seed={seed} ---", flush=True)
    folds = make_random_group_folds(seed)
    oof05 = np.full(len(y), np.nan)
    oof08 = np.full(len(y), np.nan)

    for fold_idx, (tr, va) in enumerate(folds, start=1):
        t0 = time.time()
        m05 = FinalBlendRegressor(num_threads=NUM_THREADS)
        m08 = ShrunkExpertRegressor(num_threads=NUM_THREADS)

        m05.fit(X.iloc[tr], y[tr])
        p05 = m05.predict(X.iloc[va])

        m08.fit(X.iloc[tr], y[tr])
        p08 = m08.predict(X.iloc[va])

        oof05[va] = p05
        oof08[va] = p08

        r05 = rmse(y[va], p05)
        r08 = rmse(y[va], p08)
        delta = r08 - r05
        elapsed = time.time() - t0

        rows.append({
            "repeat": repeat_idx,
            "seed": seed,
            "fold": fold_idx,
            "n_train_rows": len(tr),
            "n_valid_rows": len(va),
            "n_train_patients": int(X.iloc[tr]["patient_id"].nunique()),
            "n_valid_patients": int(X.iloc[va]["patient_id"].nunique()),
            "rmse_05": r05,
            "rmse_08": r08,
            "delta_08_minus_05": delta,
            "exp08_wins": bool(delta < 0),
            "seconds": elapsed,
        })

        print(
            f"R{repeat_idx} F{fold_idx}: "
            f"05={r05:.4f} | 08={r08:.4f} | "
            f"delta={delta:+.4f} | "
            f"{'08 WINS' if delta < 0 else '05 WINS'} | "
            f"{elapsed:.0f}s",
            flush=True,
        )

    global05 = rmse(y, oof05)
    global08 = rmse(y, oof08)
    global_delta = global08 - global05
    repeat_rows.append({
        "repeat": repeat_idx,
        "seed": seed,
        "global_rmse_05": global05,
        "global_rmse_08": global08,
        "global_delta_08_minus_05": global_delta,
    })
    print(
        f"Repeat {repeat_idx} GLOBAL: "
        f"05={global05:.4f} | 08={global08:.4f} | delta={global_delta:+.4f}",
        flush=True,
    )

fold_df = pd.DataFrame(rows)
repeat_df = pd.DataFrame(repeat_rows)

fold_path = RESULTS_DIR / "09_repeated_group_folds.csv"
repeat_path = RESULTS_DIR / "09_repeated_group_repeats.csv"
fold_df.to_csv(fold_path, index=False)
repeat_df.to_csv(repeat_path, index=False)

mean_delta = float(fold_df["delta_08_minus_05"].mean())
median_delta = float(fold_df["delta_08_minus_05"].median())
std_delta = float(fold_df["delta_08_minus_05"].std(ddof=1))
wins = int(fold_df["exp08_wins"].sum())
nfolds = len(fold_df)
win_rate = wins / nfolds

repeat_mean_delta = float(repeat_df["global_delta_08_minus_05"].mean())
repeat_min_delta = float(repeat_df["global_delta_08_minus_05"].min())
repeat_max_delta = float(repeat_df["global_delta_08_minus_05"].max())

if repeat_max_delta < 0 and win_rate >= 0.80 and repeat_mean_delta <= -0.025:
    verdict = "ROBUST_GAIN"
elif repeat_mean_delta < 0 and win_rate >= 0.60:
    verdict = "SMALL_OR_UNCERTAIN_GAIN"
else:
    verdict = "OVERFIT_SUSPECT_OR_NO_GAIN"

elapsed_total = time.time() - grand_start

lines = [
    "",
    "=" * 78,
    "FINAL ROBUSTNESS AUDIT",
    "=" * 78,
    f"Paired folds                    : {nfolds}",
    f"Exp08 wins                      : {wins}/{nfolds} ({100*win_rate:.1f}%)",
    f"Mean fold delta (08 - 05)       : {mean_delta:+.4f}",
    f"Median fold delta               : {median_delta:+.4f}",
    f"Std of fold delta               : {std_delta:.4f}",
    "",
    "Repeat-level GLOBAL deltas:",
]
for _, rr in repeat_df.iterrows():
    lines.append(
        f"  Repeat {int(rr['repeat'])} seed={int(rr['seed'])}: "
        f"05={rr['global_rmse_05']:.4f} | "
        f"08={rr['global_rmse_08']:.4f} | "
        f"delta={rr['global_delta_08_minus_05']:+.4f}"
    )

lines += [
    "",
    f"Mean repeat GLOBAL delta        : {repeat_mean_delta:+.4f}",
    f"Best/worst repeat delta         : {repeat_min_delta:+.4f} / {repeat_max_delta:+.4f}",
    f"VERDICT                         : {verdict}",
    "",
    "Decision rule fixed before looking at these results:",
    "  ROBUST_GAIN = Exp08 wins every repeat, >=80% paired folds,",
    "                and mean repeat global delta <= -0.025.",
    "  SMALL_OR_UNCERTAIN_GAIN = directionally better, but weaker stability.",
    "  OVERFIT_SUSPECT_OR_NO_GAIN = improvement does not survive new splits.",
    "",
    f"Saved fold results   : {fold_path}",
    f"Saved repeat results : {repeat_path}",
    f"Total runtime        : {elapsed_total/60:.1f} min",
    "=" * 78,
]
summary = "\n".join(lines)
print(summary)

summary_path = RESULTS_DIR / "09_repeated_group_summary.txt"
summary_path.write_text(summary, encoding="utf-8")
print(f"Saved summary        : {summary_path}")
