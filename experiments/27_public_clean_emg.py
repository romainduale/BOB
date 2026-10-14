"""Experiment 27: public clean measurements + EMG latent trajectory.

This experiment uses the public ``Kaggle_officiel`` feature files from the
Harry01530 fork.  Their rows, patients, covariates, missingness pattern and
targets align exactly with this repository; only the ON/OFF measurements are a
cleaner alternate view and ``rater_id`` is absent.  The external implementation
is pinned to a commit and every input file is hash-checked before use.

Validation is patient-disjoint GroupKFold.  Every outer model builds its stage-2
features from an inner GroupKFold, so no held-out patient target contributes to
training features.  Nothing is uploaded to Skore or Kaggle by this script.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp27"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

REFERENCE_URL = "https://github.com/Harry01530/bobathon-esilv.git"
REFERENCE_COMMIT = "a608a6f86a0f7886f46982bf19535fcfb8b99cd6"
REFERENCE_ROOT = Path("/private/tmp") / f"harry-bobathon-{REFERENCE_COMMIT[:12]}"
EXPECTED_HASHES = {
    "X_train.csv": "ae117940d28ab480a7e6f0da5bc535f6175229653459f5ce6ffabda0cf7e9534",
    "X_test.csv": "33b3906ba1847567dab3a1a08d3bb2250010fe6c30630cbdc63a716696251d36",
    "y_train.csv": "8d8583da6899fa8cbe58b1dd522693f24211da93f6b376ae42ba471394743775",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_reference_checkout() -> Path:
    if not REFERENCE_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", REFERENCE_URL, str(REFERENCE_ROOT)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(REFERENCE_ROOT), "checkout", REFERENCE_COMMIT],
            check=True,
        )
    actual_commit = subprocess.run(
        ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != REFERENCE_COMMIT:
        raise RuntimeError(
            f"Reference checkout is {actual_commit}, expected {REFERENCE_COMMIT}"
        )
    return REFERENCE_ROOT


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    reference_root = ensure_reference_checkout()
    reference_data = reference_root / "data" / "Kaggle_officiel"
    for filename, expected in EXPECTED_HASHES.items():
        path = reference_data / filename if filename.startswith("X_") else DATA_DIR / filename
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Unexpected SHA256 for {path}: {actual}")

    sys.path.insert(0, str(reference_root / "src"))
    from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
    from parkinsons.two_stage_mle import TwoStageMLE

    train = pd.read_csv(reference_data / "X_train.csv")
    test = pd.read_csv(reference_data / "X_test.csv")
    local_train = pd.read_csv(DATA_DIR / "X_train.csv")
    local_test = pd.read_csv(DATA_DIR / "X_test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y_frame = pd.read_csv(DATA_DIR / "y_train.csv").set_index("Index")

    for clean, local, split in [
        (train, local_train, "train"),
        (test, local_test, "test"),
    ]:
        if not clean["Index"].equals(local["Index"]):
            raise RuntimeError(f"Index mismatch in {split}")
        if not clean["patient_id"].equals(local["patient_id"]):
            raise RuntimeError(f"patient_id mismatch in {split}")
        for column in clean.columns.difference(["on", "off"]):
            if not clean[column].equals(local[column]):
                raise RuntimeError(f"Unexpected {column} mismatch in {split}")

    y = y_frame.loc[train["Index"], "target"].to_numpy(dtype=float)
    groups = train["patient_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(train, y, groups=groups))
    for fold_number, (fit_indices, validation_indices) in enumerate(folds, start=1):
        overlap = set(groups[fit_indices]) & set(groups[validation_indices])
        if overlap:
            raise RuntimeError(f"Patient leakage in fold {fold_number}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(train), np.nan)
    test_predictions = []
    fold_scores = []

    for fold_number, (fit_indices, validation_indices) in enumerate(folds, start=1):
        cache_path = RESULTS_DIR / f"fold_{fold_number}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            if not np.array_equal(cached["validation_indices"], validation_indices):
                raise RuntimeError(f"Stale cache indices in fold {fold_number}")
            validation_prediction = cached["validation_prediction"]
            test_prediction = cached["test_prediction"]
            print(f"Fold {fold_number}/5 loaded from cache", flush=True)
        else:
            fit_frame = train.iloc[fit_indices].reset_index(drop=True)
            validation_frame = train.iloc[validation_indices].reset_index(drop=True)
            category_reference = pd.concat(
                [validation_frame, test], ignore_index=True
            )
            cat_dtypes = make_cat_dtypes(fit_frame, category_reference)
            model = TwoStageMLE(
                cat_dtypes=cat_dtypes,
                inner_n_splits=5,
                bias_params={**BIAS_PARAMS, "prior_scale": 0.3},
                smooth="quad_iso",
                two_stage=True,
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
            )
            model.fit(fit_frame, y[fit_indices])
            validation_prediction = model.predict(validation_frame)
            test_prediction = model.predict(test.reset_index(drop=True))
            np.savez_compressed(
                cache_path,
                validation_indices=validation_indices,
                validation_prediction=validation_prediction,
                test_prediction=test_prediction,
            )

        oof[validation_indices] = validation_prediction
        test_predictions.append(test_prediction)
        score = rmse(y[validation_indices], validation_prediction)
        fold_scores.append(score)
        print(f"Fold {fold_number}/5 RMSE={score:.4f}", flush=True)

    if np.isnan(oof).any():
        raise RuntimeError("Missing OOF predictions")

    test_mean = np.mean(np.column_stack(test_predictions), axis=1)
    submission = (
        pd.DataFrame({"Index": test["Index"], "target": test_mean})
        .set_index("Index")
        .loc[sample["Index"]]
        .reset_index()
    )
    if list(submission.columns) != ["Index", "target"]:
        raise RuntimeError("Invalid submission columns")
    if len(submission) != 11013 or submission["target"].isna().any():
        raise RuntimeError("Invalid submission rows")

    # Independent patient assignment for a stability check.  Hyperparameters
    # remain fixed; this split is never used to choose or blend the submission.
    patient_ids = np.sort(train["patient_id"].unique())
    rng = np.random.default_rng(7)
    rng.shuffle(patient_ids)
    patient_order = {
        patient_id: position for position, patient_id in enumerate(patient_ids)
    }
    shuffled_groups = train["patient_id"].map(patient_order).to_numpy()
    second_folds = list(
        GroupKFold(n_splits=5).split(train, y, groups=shuffled_groups)
    )
    second_oof = np.full(len(train), np.nan)
    second_fold_scores = []
    for fold_number, (fit_indices, validation_indices) in enumerate(
        second_folds, start=1
    ):
        cache_path = RESULTS_DIR / f"split2_fold_{fold_number}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            if not np.array_equal(cached["validation_indices"], validation_indices):
                raise RuntimeError(f"Stale second-split cache in fold {fold_number}")
            prediction = cached["prediction"]
            print(f"Second split fold {fold_number}/5 loaded from cache", flush=True)
        else:
            fit_frame = train.iloc[fit_indices].reset_index(drop=True)
            validation_frame = train.iloc[validation_indices].reset_index(drop=True)
            cat_dtypes = make_cat_dtypes(fit_frame, validation_frame)
            model = TwoStageMLE(
                cat_dtypes=cat_dtypes,
                inner_n_splits=5,
                bias_params={**BIAS_PARAMS, "prior_scale": 0.3},
                smooth="quad_iso",
                two_stage=True,
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
            )
            model.fit(fit_frame, y[fit_indices])
            prediction = model.predict(validation_frame)
            np.savez_compressed(
                cache_path,
                validation_indices=validation_indices,
                prediction=prediction,
            )
        second_oof[validation_indices] = prediction
        second_score = rmse(y[validation_indices], prediction)
        second_fold_scores.append(second_score)
        print(
            f"Second split fold {fold_number}/5 RMSE={second_score:.4f}",
            flush=True,
        )

    if np.isnan(second_oof).any():
        raise RuntimeError("Missing second-split OOF predictions")

    np.save(RESULTS_DIR / "oof.npy", oof)
    np.save(RESULTS_DIR / "oof_split2.npy", second_oof)
    np.save(RESULTS_DIR / "test_mean.npy", test_mean)
    submission_path = SUBMISSIONS_DIR / "27_public_clean_emg_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    summary = {
        "model": "TwoStageMLE with EMG bias likelihood and quadratic-isotonic smoothing",
        "external_data": "Harry01530 public Kaggle_officiel X_train/X_test",
        "reference_url": REFERENCE_URL,
        "reference_commit": REFERENCE_COMMIT,
        "validation": "5-fold patient GroupKFold with 5-fold inner OOF",
        "global_oof_rmse": rmse(y, oof),
        "fold_rmse": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "second_split": {
            "seed": 7,
            "global_oof_rmse": rmse(y, second_oof),
            "fold_rmse": second_fold_scores,
            "fold_mean": float(np.mean(second_fold_scores)),
            "fold_std": float(np.std(second_fold_scores)),
            "delta_vs_primary": abs(rmse(y, second_oof) - rmse(y, oof)),
        },
        "submission": str(submission_path.relative_to(PROJECT_ROOT)),
        "submission_rows": len(submission),
        "submission_min": float(submission["target"].min()),
        "submission_max": float(submission["target"].max()),
        "submission_mean": float(submission["target"].mean()),
        "caveat": "Score uses a cleaner public alternate feature view and is not comparable to noisy-only CV.",
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
