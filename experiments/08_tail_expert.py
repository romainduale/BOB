"""Experiment 08: lightweight residual expert on top of the Exp-07 winner.

This experiment deliberately keeps the search space tiny.  It compares the
clipped Exp-07 OOF predictions with one fixed HistGradientBoosting residual
expert.  The expert only consumes features available at inference time.

The script does not submit anything to Kaggle and does not push to GitHub.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp08"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(exist_ok=True)

LOWER_TARGET = 0.0
UPPER_TARGET = 132.0
SEVERITY_THRESHOLD = 50.0

EXPERT_PARAMS = {
    "max_iter": 80,
    "learning_rate": 0.04,
    "max_depth": 2,
    "min_samples_leaf": 80,
    "l2_regularization": 10.0,
    "random_state": 42,
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clipped(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, LOWER_TARGET, UPPER_TARGET)


def category_maps(frame: pd.DataFrame) -> dict[str, dict[object, int]]:
    maps: dict[str, dict[object, int]] = {}
    for column in ["cohort", "gene", "rater_id", "sexM"]:
        values = sorted(frame[column].dropna().unique().tolist(), key=str)
        maps[column] = {value: index for index, value in enumerate(values)}
    return maps


def build_meta_features(
    raw: pd.DataFrame,
    base_prediction: np.ndarray,
    maps: dict[str, dict[object, int]],
) -> pd.DataFrame:
    """Create target-free row and patient features for the residual expert."""
    features = pd.DataFrame(index=raw.index)
    features["base_prediction"] = base_prediction
    features["base_prediction_sq"] = base_prediction**2 / 100.0
    features["base_above_50"] = np.maximum(base_prediction - 50.0, 0.0)
    features["base_above_60"] = np.maximum(base_prediction - 60.0, 0.0)

    numeric = [
        "age",
        "age_at_diagnosis",
        "ledd",
        "time_since_intake_on",
        "time_since_intake_off",
        "on",
        "off",
    ]
    for column in numeric:
        features[column] = raw[column]
        features[f"{column}_missing"] = raw[column].isna().astype("int8")

    features["disease_duration"] = raw["age"] - raw["age_at_diagnosis"]
    features["off_minus_on"] = raw["off"] - raw["on"]
    features["regime"] = (
        raw["off"].notna().astype("int8")
        + 2 * raw["on"].notna().astype("int8")
    )

    for column, mapping in maps.items():
        features[f"{column}_code"] = raw[column].map(mapping).fillna(-1).astype(int)

    patient_frame = pd.DataFrame(
        {
            "patient_id": raw["patient_id"].to_numpy(),
            "base": base_prediction,
            "on": raw["on"].to_numpy(),
            "off": raw["off"].to_numpy(),
        },
        index=raw.index,
    )
    grouped = patient_frame.groupby("patient_id", sort=False)
    for column in ["base", "on", "off"]:
        for statistic in ["mean", "min", "max", "std"]:
            features[f"{column}_patient_{statistic}"] = grouped[column].transform(
                statistic
            )
    features["n_visits"] = grouped["base"].transform("size")
    return features


def primary_folds(
    raw: pd.DataFrame, target: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(
        GroupKFold(n_splits=5).split(
            raw, target, groups=raw["patient_id"].to_numpy()
        )
    )


def shuffled_patient_folds(
    raw: pd.DataFrame, target: np.ndarray, seed: int = 7
) -> list[tuple[np.ndarray, np.ndarray]]:
    patient_ids = sorted(raw["patient_id"].unique().tolist())
    random.Random(seed).shuffle(patient_ids)
    order = {patient_id: position for position, patient_id in enumerate(patient_ids)}
    shuffled_groups = raw["patient_id"].map(order).to_numpy()
    return list(GroupKFold(n_splits=5).split(raw, target, groups=shuffled_groups))


def cross_fitted_expert(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    target: np.ndarray,
    base_prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    output = np.full(len(target), np.nan)
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_patients = set(raw.iloc[train_indices]["patient_id"])
        validation_patients = set(raw.iloc[validation_indices]["patient_id"])
        assert not train_patients.intersection(validation_patients)

        model = HistGradientBoostingRegressor(**EXPERT_PARAMS)
        model.fit(
            features.iloc[train_indices],
            target[train_indices] - base_prediction[train_indices],
        )
        correction = model.predict(features.iloc[validation_indices])
        output[validation_indices] = clipped(
            base_prediction[validation_indices] + correction
        )
        print(
            f"  fold {fold_number}: "
            f"baseline={rmse(target[validation_indices], clipped(base_prediction[validation_indices])):.4f} "
            f"expert={rmse(target[validation_indices], output[validation_indices]):.4f}",
            flush=True,
        )
    assert not np.isnan(output).any()
    return output


def range_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, dict]:
    ranges = {
        "le_10": target <= 10,
        "10_25": (target > 10) & (target <= 25),
        "25_50": (target > 25) & (target <= 50),
        "50_70": (target > 50) & (target <= 70),
        "gt_70": target > 70,
    }
    output: dict[str, dict] = {}
    squared_errors = (target - prediction) ** 2
    for label, mask in ranges.items():
        output[label] = {
            "n": int(mask.sum()),
            "rmse": rmse(target[mask], prediction[mask]),
            "bias_y_minus_pred": float(np.mean(target[mask] - prediction[mask])),
            "mse_share": float(squared_errors[mask].sum() / squared_errors.sum()),
        }
    return output


print("Loading Exp-07 predictions...", flush=True)
X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy()
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="Index")

base_oof = np.load(PROJECT_ROOT / "results" / "exp07" / "oof_winner.npy")
base_submission_path = SUBMISSIONS_DIR / "07_winner_honest.csv"
base_test_frame = pd.read_csv(base_submission_path).set_index("Index")
base_test = base_test_frame.loc[X_test.index, "target"].to_numpy()

assert len(base_oof) == len(y_train)
assert list(X_test.index) == list(sample_submission.index)

maps = category_maps(X_train)
train_features = build_meta_features(X_train, base_oof, maps)
test_features = build_meta_features(X_test, base_test, maps)
test_features = test_features.reindex(columns=train_features.columns)

folds_primary = primary_folds(X_train, y_train)
folds_secondary = shuffled_patient_folds(X_train, y_train, seed=7)

baseline_oof = clipped(base_oof)
print(f"Clipped Exp-07 baseline: {rmse(y_train, baseline_oof):.4f}")

print("\nPrimary patient-grouped validation", flush=True)
expert_oof = cross_fitted_expert(
    X_train, train_features, y_train, base_oof, folds_primary
)
print("\nSecondary patient-grouped validation", flush=True)
expert_oof_secondary = cross_fitted_expert(
    X_train, train_features, y_train, base_oof, folds_secondary
)

primary_fold_scores = [
    rmse(y_train[validation], expert_oof[validation])
    for _, validation in folds_primary
]
secondary_fold_scores = [
    rmse(y_train[validation], expert_oof_secondary[validation])
    for _, validation in folds_secondary
]

baseline_rmse = rmse(y_train, baseline_oof)
expert_rmse = rmse(y_train, expert_oof)
secondary_rmse = rmse(y_train, expert_oof_secondary)
normal = y_train < SEVERITY_THRESHOLD
severe = ~normal

summary = {
    "model": "Exp-08 residual HGBR expert on Exp-07",
    "baseline_clipped_rmse": baseline_rmse,
    "primary_oof_rmse": expert_rmse,
    "gain_vs_clipped_exp07": baseline_rmse - expert_rmse,
    "primary_fold_rmses": primary_fold_scores,
    "secondary_oof_rmse": secondary_rmse,
    "secondary_fold_rmses": secondary_fold_scores,
    "normal_lt_50": {
        "baseline_rmse": rmse(y_train[normal], baseline_oof[normal]),
        "expert_rmse": rmse(y_train[normal], expert_oof[normal]),
    },
    "severe_ge_50": {
        "baseline_rmse": rmse(y_train[severe], baseline_oof[severe]),
        "expert_rmse": rmse(y_train[severe], expert_oof[severe]),
    },
    "target_ranges": range_metrics(y_train, expert_oof),
    "expert_parameters": EXPERT_PARAMS,
    "clipping": [LOWER_TARGET, UPPER_TARGET],
}

print("\nFitting final residual expert...", flush=True)
final_expert = HistGradientBoostingRegressor(**EXPERT_PARAMS)
final_expert.fit(train_features, y_train - base_oof)
test_prediction = clipped(base_test + final_expert.predict(test_features))

assert len(test_prediction) == len(sample_submission)
assert not np.isnan(test_prediction).any()
assert list(X_test.index) == list(sample_submission.index)

submission_path = SUBMISSIONS_DIR / "08_tail_expert.csv"
pd.DataFrame({"Index": X_test.index, "target": test_prediction}).to_csv(
    submission_path, index=False
)

np.save(RESULTS_DIR / "oof_tail_expert.npy", expert_oof)
np.save(RESULTS_DIR / "oof_tail_expert_split2.npy", expert_oof_secondary)
pd.DataFrame(
    {
        "Index": X_train.index,
        "patient_id": X_train["patient_id"],
        "y_true": y_train,
        "exp07_clipped": baseline_oof,
        "exp08_tail_expert": expert_oof,
    }
).to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
(RESULTS_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)

print("\n=== EXP-08 SUMMARY ===")
print(f"Exp-07 clipped OOF : {baseline_rmse:.4f}")
print(f"Exp-08 primary OOF : {expert_rmse:.4f}")
print(f"Gain               : {baseline_rmse - expert_rmse:+.4f}")
print(f"Primary folds      : {[round(value, 4) for value in primary_fold_scores]}")
print(f"Secondary OOF      : {secondary_rmse:.4f}")
print(f"Secondary folds    : {[round(value, 4) for value in secondary_fold_scores]}")
print(
    f"Normal <50        : {summary['normal_lt_50']['baseline_rmse']:.4f} -> "
    f"{summary['normal_lt_50']['expert_rmse']:.4f}"
)
print(
    f"Severe >=50       : {summary['severe_ge_50']['baseline_rmse']:.4f} -> "
    f"{summary['severe_ge_50']['expert_rmse']:.4f}"
)
print(
    f"Submission        : {submission_path} "
    f"(rows={len(test_prediction)}, min={test_prediction.min():.3f}, "
    f"max={test_prediction.max():.3f}, NaN={np.isnan(test_prediction).sum()})"
)
