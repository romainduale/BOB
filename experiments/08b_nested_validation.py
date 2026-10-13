"""Strict nested validation for the fixed Exp-08 residual expert.

For every outer patient fold, the regime HGBR and residual CatBoost components
are rebuilt from scratch. Inner patient-grouped OOF predictions are then used
to learn the Exp-07 post-processing chain and the fixed Exp-08 residual expert.
The outer validation patients are never used to fit either layer.

This script validates only. It does not create a Kaggle submission or contact
Skore/GitHub.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import nnls
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp08"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NULLABLE = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]
CAT_COLUMNS = ["cohort", "gene", "rater_id", "sexM"]

HGBR_PARAMS = {
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_samples_leaf": 10,
    "max_bins": 63,
    "random_state": RANDOM_STATE,
}
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "thread_count": -1,
    "allow_writing_files": False,
}
EXPERT_PARAMS = {
    "max_iter": 80,
    "learning_rate": 0.04,
    "max_depth": 2,
    "min_samples_leaf": 80,
    "l2_regularization": 10.0,
    "random_state": RANDOM_STATE,
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clip(prediction: np.ndarray) -> np.ndarray:
    return np.clip(prediction, 0.0, 132.0)


def regimes(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    off = raw["off"].to_numpy()
    on = raw["on"].to_numpy()
    return {
        "both": (~np.isnan(off)) & (~np.isnan(on)),
        "off_only": (~np.isnan(off)) & np.isnan(on),
        "on_only": np.isnan(off) & (~np.isnan(on)),
    }


def _poly_coefficients(ages: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    if len(ages) >= 3:
        try:
            return tuple(float(value) for value in np.polyfit(ages, values, 2))
        except Exception:
            pass
    if len(ages) >= 2:
        try:
            slope, intercept = np.polyfit(ages, values, 1)
            return 0.0, float(slope), float(intercept)
        except Exception:
            pass
    return 0.0, 0.0, float(np.mean(values))


def engineer_hgbr(raw: pd.DataFrame, mappings: dict[str, dict[object, int]]) -> pd.DataFrame:
    """Exp-05 regime features, computed from each patient's own X visits only."""
    output = raw.copy()
    output["disease_duration"] = output["age"] - output["age_at_diagnosis"]
    output["off_minus_on"] = output["off"] - output["on"]
    output["regime"] = output["off"].notna().astype(int) + 2 * output["on"].notna().astype(int)
    for column in NULLABLE:
        output[f"{column}_missing"] = output[column].isna().astype(float)

    grouped = raw.groupby("patient_id", sort=False)
    age_stats = grouped["age"].agg(
        patient_age_mean="mean", patient_age_min="min", patient_age_max="max"
    )
    age_stats["patient_age_span"] = age_stats["patient_age_max"] - age_stats["patient_age_min"]
    output = output.join(age_stats, on="patient_id")
    output["n_visits"] = grouped.size().reindex(output["patient_id"]).to_numpy()

    counts = pd.DataFrame(index=age_stats.index)
    for column in ["off", "on", "ledd"]:
        counts[f"n_{column}"] = grouped[column].count()
        counts[f"frac_{column}"] = counts[f"n_{column}"] / grouped.size()
    output = output.join(counts, on="patient_id")

    differences = raw.groupby("patient_id", sort=False).apply(
        lambda frame: pd.Series(
            {
                "off_on_diff_mean": (frame["off"] - frame["on"]).mean(),
                "off_on_diff_std": (frame["off"] - frame["on"]).std(),
            }
        ),
        include_groups=False,
    )
    output = output.join(differences, on="patient_id")

    timing = grouped["age"].agg(patient_first_age="min", patient_last_age="max")
    timing["patient_followup"] = timing["patient_last_age"] - timing["patient_first_age"]
    output = output.join(timing, on="patient_id")

    for column in NULLABLE:
        statistics = grouped[column].agg(
            **{
                f"{column}_patient_mean": "mean",
                f"{column}_patient_median": "median",
                f"{column}_patient_min": "min",
                f"{column}_patient_max": "max",
                f"{column}_patient_std": "std",
                f"{column}_patient_q25": lambda values: values.quantile(0.25),
                f"{column}_patient_q75": lambda values: values.quantile(0.75),
                f"{column}_patient_count": "count",
            }
        )
        output = output.join(statistics, on="patient_id")
        output[f"{column}_patient_observed_fraction"] = (
            output[f"{column}_patient_count"] / output["n_visits"]
        )

    output["age_relative_mean"] = output["age"] - output["patient_age_mean"]
    output["age_relative_min"] = output["age"] - output["patient_age_min"]
    output["age_relative_max"] = output["age"] - output["patient_age_max"]
    output["age_relative_first"] = output["age"] - output["patient_first_age"]
    output["visit_rank_ascending"] = grouped["age"].rank(method="dense")
    output["visit_rank_descending"] = grouped["age"].rank(method="dense", ascending=False)
    ordered = output.sort_values(["patient_id", "age"])
    output.loc[ordered.index, "age_difference_previous"] = ordered.groupby("patient_id")["age"].diff()
    output.loc[ordered.index, "age_difference_next"] = ordered.groupby("patient_id")["age"].diff(-1).abs()

    for column in ["on", "off", "ledd"]:
        slope = pd.Series(np.nan, index=output.index)
        intercept = pd.Series(np.nan, index=output.index)
        fitted_linear = pd.Series(np.nan, index=output.index)
        fitted_quadratic = pd.Series(np.nan, index=output.index)
        for patient_id, frame in raw.groupby("patient_id", sort=False):
            valid = frame[["age", column]].dropna()
            patient_index = frame.index
            ages = frame["age"].to_numpy()
            if len(valid) >= 2 and valid["age"].nunique() > 1:
                fit = linregress(valid["age"].to_numpy(), valid[column].to_numpy())
                slope.loc[patient_index] = fit.slope
                intercept.loc[patient_index] = fit.intercept
                fitted_linear.loc[patient_index] = fit.intercept + fit.slope * ages
                a, b, c = _poly_coefficients(valid["age"].to_numpy(), valid[column].to_numpy())
                fitted_quadratic.loc[patient_index] = a * ages**2 + b * ages + c
            elif len(valid) == 1:
                value = float(valid[column].iloc[0])
                fitted_linear.loc[patient_index] = value
                fitted_quadratic.loc[patient_index] = value
        output[f"{column}_patient_slope"] = slope
        output[f"{column}_patient_intercept"] = intercept
        output[f"{column}_patient_fitted_linear"] = fitted_linear
        output[f"{column}_patient_fitted_quadratic"] = fitted_quadratic
        output[f"{column}_residual_linear"] = output[column] - fitted_linear

    for column, mapping in mappings.items():
        output[column] = output[column].map(mapping).fillna(-1).astype(int)
    return output.drop(columns=["patient_id"])


def engineer_catboost(raw: pd.DataFrame) -> pd.DataFrame:
    output = raw.copy()
    output["disease_duration"] = output["age"] - output["age_at_diagnosis"]
    output["off_minus_on"] = output["off"] - output["on"]
    for column in NULLABLE:
        output[f"{column}_missing"] = output[column].isna().astype(float)
    grouped = raw.groupby("patient_id", sort=False)
    for column in NULLABLE:
        stats = grouped[column].agg(
            **{
                f"{column}_patient_mean": "mean",
                f"{column}_patient_std": "std",
                f"{column}_patient_count": "count",
            }
        )
        output = output.join(stats, on="patient_id")
    ages = grouped["age"].agg(patient_age_mean="mean", patient_age_min="min")
    output = output.join(ages, on="patient_id")
    output["n_visits"] = grouped.size().reindex(output["patient_id"]).to_numpy()
    output["age_relative_mean"] = output["age"] - output["patient_age_mean"]
    output["visit_rank_ascending"] = grouped["age"].rank(method="dense")
    for column in ["on", "off", "ledd"]:
        slope = pd.Series(np.nan, index=output.index)
        fitted = pd.Series(np.nan, index=output.index)
        for _, frame in raw.groupby("patient_id", sort=False):
            valid = frame[["age", column]].dropna()
            if len(valid) >= 2 and valid["age"].nunique() > 1:
                fit = linregress(valid["age"].to_numpy(), valid[column].to_numpy())
                slope.loc[frame.index] = fit.slope
                fitted.loc[frame.index] = fit.intercept + fit.slope * frame["age"].to_numpy()
            elif len(valid) == 1:
                fitted.loc[frame.index] = float(valid[column].iloc[0])
        output[f"{column}_patient_slope"] = slope
        output[f"{column}_patient_fitted"] = fitted
    output = output.drop(columns=["patient_id"])
    for column in CAT_COLUMNS:
        output[column] = output[column].fillna("missing").astype(str)
    return output


def categorical_mappings(*frames: pd.DataFrame) -> dict[str, dict[object, int]]:
    combined = pd.concat([frame[CAT_COLUMNS] for frame in frames], axis=0)
    return {
        column: {
            value: position
            for position, value in enumerate(
                sorted(combined[column].dropna().unique().tolist(), key=str)
            )
        }
        for column in CAT_COLUMNS
    }


def fit_predict_components(
    train_raw: pd.DataFrame,
    train_target: np.ndarray,
    validation_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the two Exp-07 base components and predict unseen patients."""
    # Fit category encodings on the training partition only. Unseen validation
    # values are represented by -1 in ``engineer_hgbr``.
    mappings = categorical_mappings(train_raw)
    train_hgbr = engineer_hgbr(train_raw, mappings)
    validation_hgbr = engineer_hgbr(validation_raw, mappings)
    common_hgbr = train_hgbr.columns.intersection(validation_hgbr.columns).tolist()
    train_hgbr = train_hgbr[common_hgbr]
    validation_hgbr = validation_hgbr[common_hgbr]

    train_regimes = regimes(train_raw)
    validation_regimes = regimes(validation_raw)
    regime_prediction = np.full(len(validation_raw), np.nan)
    for name in ["both", "off_only", "on_only"]:
        train_mask = train_regimes[name]
        validation_mask = validation_regimes[name]
        if not validation_mask.any():
            continue
        keep = [
            column
            for column in train_hgbr.columns
            if train_hgbr.loc[train_mask, column].nunique(dropna=False) > 1
        ]
        model = HistGradientBoostingRegressor(**HGBR_PARAMS)
        if name in {"both", "off_only"}:
            anchor_train = train_raw.loc[train_mask, "off"].to_numpy()
            anchor_validation = validation_raw.loc[validation_mask, "off"].to_numpy()
        else:
            anchor_train = train_raw.loc[train_mask, "on"].to_numpy()
            anchor_validation = validation_raw.loc[validation_mask, "on"].to_numpy()
        model.fit(train_hgbr.loc[train_mask, keep], train_target[train_mask] - anchor_train)
        regime_prediction[validation_mask] = anchor_validation + model.predict(
            validation_hgbr.loc[validation_mask, keep]
        )

    train_cat = engineer_catboost(train_raw)
    validation_cat = engineer_catboost(validation_raw)
    common_cat = train_cat.columns.intersection(validation_cat.columns).tolist()
    categorical_indices = [
        position for position, column in enumerate(common_cat) if column in CAT_COLUMNS
    ]
    off_train = train_raw["off"].to_numpy()
    off_validation = validation_raw["off"].to_numpy()
    observed_train = ~np.isnan(off_train)
    observed_validation = ~np.isnan(off_validation)
    missing_train = ~observed_train
    missing_validation = ~observed_validation
    cat_prediction = np.full(len(validation_raw), np.nan)

    residual_model = CatBoostRegressor(**CATBOOST_PARAMS)
    residual_model.fit(
        Pool(
            train_cat.loc[observed_train, common_cat],
            train_target[observed_train] - off_train[observed_train],
            cat_features=categorical_indices,
        )
    )
    if observed_validation.any():
        cat_prediction[observed_validation] = off_validation[observed_validation] + residual_model.predict(
            Pool(
                validation_cat.loc[observed_validation, common_cat],
                cat_features=categorical_indices,
            )
        )

    direct_model = CatBoostRegressor(**CATBOOST_PARAMS)
    direct_model.fit(
        Pool(
            train_cat.loc[missing_train, common_cat],
            train_target[missing_train],
            cat_features=categorical_indices,
        )
    )
    if missing_validation.any():
        cat_prediction[missing_validation] = direct_model.predict(
            Pool(
                validation_cat.loc[missing_validation, common_cat],
                cat_features=categorical_indices,
            )
        )

    assert not np.isnan(regime_prediction).any()
    assert not np.isnan(cat_prediction).any()
    return regime_prediction, cat_prediction


def smooth(prediction: np.ndarray, raw: pd.DataFrame) -> np.ndarray:
    result = prediction.copy()
    patient_ids = raw["patient_id"].to_numpy()
    ages = raw["age"].to_numpy()
    for patient_id in np.unique(patient_ids):
        mask = patient_ids == patient_id
        if mask.sum() < 2:
            continue
        degree = min(2, mask.sum() - 1)
        coefficients = np.polyfit(ages[mask], prediction[mask], degree)
        fitted = np.polyval(coefficients, ages[mask])
        shrink = 0.3 if mask.sum() <= 3 else 0.0
        result[mask] = (1.0 - shrink) * fitted + shrink * prediction[mask]
    return result


def rater_errors(target: np.ndarray, prediction: np.ndarray, raw: pd.DataFrame) -> dict[object, float]:
    raters = raw["rater_id"].to_numpy()
    residual = target - prediction
    return {
        rater: float(residual[raters == rater].mean())
        for rater in np.unique(raters)
    }


def apply_rater(prediction: np.ndarray, raw: pd.DataFrame, errors: dict[object, float]) -> np.ndarray:
    return prediction + 0.9 * np.array(
        [errors.get(rater, 0.0) for rater in raw["rater_id"].to_numpy()]
    )


def rater_cross_fit(
    target: np.ndarray,
    prediction: np.ndarray,
    raw: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    output = np.empty(len(target))
    for train_indices, validation_indices in folds:
        errors = rater_errors(
            target[train_indices], prediction[train_indices], raw.iloc[train_indices]
        )
        output[validation_indices] = apply_rater(
            prediction[validation_indices], raw.iloc[validation_indices], errors
        )
    return output


def normalized_nnls(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    weights, _ = nnls(matrix, target)
    if weights.sum() <= 1e-12:
        return np.ones(matrix.shape[1]) / matrix.shape[1]
    return weights / weights.sum()


def blend_cross_fit(
    target: np.ndarray,
    components: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    output = np.empty(len(target))
    for train_indices, validation_indices in folds:
        weights = normalized_nnls(components[train_indices], target[train_indices])
        output[validation_indices] = components[validation_indices] @ weights
    return output


def build_meta_features(
    raw: pd.DataFrame,
    base_prediction: np.ndarray,
    mappings: dict[str, dict[object, int]],
) -> pd.DataFrame:
    features = pd.DataFrame(index=raw.index)
    features["base_prediction"] = base_prediction
    features["base_prediction_sq"] = base_prediction**2 / 100.0
    features["base_above_50"] = np.maximum(base_prediction - 50.0, 0.0)
    features["base_above_60"] = np.maximum(base_prediction - 60.0, 0.0)
    for column in [
        "age",
        "age_at_diagnosis",
        "ledd",
        "time_since_intake_on",
        "time_since_intake_off",
        "on",
        "off",
    ]:
        features[column] = raw[column]
        features[f"{column}_missing"] = raw[column].isna().astype("int8")
    features["disease_duration"] = raw["age"] - raw["age_at_diagnosis"]
    features["off_minus_on"] = raw["off"] - raw["on"]
    features["regime"] = raw["off"].notna().astype(int) + 2 * raw["on"].notna().astype(int)
    for column, mapping in mappings.items():
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
            features[f"{column}_patient_{statistic}"] = grouped[column].transform(statistic)
    features["n_visits"] = grouped["base"].transform("size")
    return features


def nested_outer_prediction(
    outer_train_raw: pd.DataFrame,
    outer_train_target: np.ndarray,
    outer_validation_raw: pd.DataFrame,
    inner_splits: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    inner_folds = list(
        GroupKFold(n_splits=inner_splits).split(
            outer_train_raw,
            outer_train_target,
            groups=outer_train_raw["patient_id"].to_numpy(),
        )
    )
    regime_oof = np.empty(len(outer_train_raw))
    cat_oof = np.empty(len(outer_train_raw))
    for inner_number, (inner_train, inner_validation) in enumerate(inner_folds, start=1):
        print(f"      inner fold {inner_number}/{inner_splits}", flush=True)
        regime_oof[inner_validation], cat_oof[inner_validation] = fit_predict_components(
            outer_train_raw.iloc[inner_train],
            outer_train_target[inner_train],
            outer_train_raw.iloc[inner_validation],
        )

    regime_sm = smooth(regime_oof, outer_train_raw)
    cat_sm = smooth(cat_oof, outer_train_raw)
    regime_sm_rc = rater_cross_fit(outer_train_target, regime_sm, outer_train_raw, inner_folds)
    cat_sm_rc = rater_cross_fit(outer_train_target, cat_sm, outer_train_raw, inner_folds)
    component_oof = np.column_stack([regime_sm_rc, cat_sm_rc])
    blend_oof = blend_cross_fit(outer_train_target, component_oof, inner_folds)
    blend_sm_oof = smooth(blend_oof, outer_train_raw)
    winner_oof = rater_cross_fit(
        outer_train_target, blend_sm_oof, outer_train_raw, inner_folds
    )

    print("      fitting outer-train base components", flush=True)
    regime_validation, cat_validation = fit_predict_components(
        outer_train_raw, outer_train_target, outer_validation_raw
    )
    regime_validation_sm = smooth(regime_validation, outer_validation_raw)
    cat_validation_sm = smooth(cat_validation, outer_validation_raw)
    regime_validation_rc = apply_rater(
        regime_validation_sm,
        outer_validation_raw,
        rater_errors(outer_train_target, regime_sm, outer_train_raw),
    )
    cat_validation_rc = apply_rater(
        cat_validation_sm,
        outer_validation_raw,
        rater_errors(outer_train_target, cat_sm, outer_train_raw),
    )
    weights = normalized_nnls(component_oof, outer_train_target)
    blend_validation = np.column_stack([regime_validation_rc, cat_validation_rc]) @ weights
    blend_validation_sm = smooth(blend_validation, outer_validation_raw)
    winner_validation = apply_rater(
        blend_validation_sm,
        outer_validation_raw,
        rater_errors(outer_train_target, blend_sm_oof, outer_train_raw),
    )
    return winner_oof, winner_validation


def outer_patient_folds(
    raw: pd.DataFrame, target: np.ndarray, fold_seed: int | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = raw["patient_id"].to_numpy()
    if fold_seed is not None:
        patient_ids = sorted(raw["patient_id"].unique().tolist())
        random.Random(fold_seed).shuffle(patient_ids)
        order = {
            patient_id: position for position, patient_id in enumerate(patient_ids)
        }
        groups = raw["patient_id"].map(order).to_numpy()
    return list(GroupKFold(n_splits=5).split(raw, target, groups=groups))


def run_nested_validation(fold_seed: int | None = None) -> dict:
    X = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy()
    outer_folds = outer_patient_folds(X, y, fold_seed)
    nested_base = np.empty(len(y))
    nested_expert = np.empty(len(y))

    for outer_number, (outer_train, outer_validation) in enumerate(
        outer_folds, start=1
    ):
        print(f"\n=== OUTER FOLD {outer_number}/5 ===", flush=True)
        X_train = X.iloc[outer_train]
        X_validation = X.iloc[outer_validation]
        y_train = y[outer_train]
        assert not set(X_train["patient_id"]).intersection(
            X_validation["patient_id"]
        )

        inner_oof, validation_base = nested_outer_prediction(
            X_train, y_train, X_validation, inner_splits=3
        )
        mappings = categorical_mappings(X_train)
        meta_train = build_meta_features(X_train, inner_oof, mappings)
        meta_validation = build_meta_features(X_validation, validation_base, mappings)
        meta_validation = meta_validation.reindex(columns=meta_train.columns)

        expert = HistGradientBoostingRegressor(**EXPERT_PARAMS)
        expert.fit(meta_train, y_train - inner_oof)
        validation_expert = clip(validation_base + expert.predict(meta_validation))
        nested_base[outer_validation] = clip(validation_base)
        nested_expert[outer_validation] = validation_expert
        print(
            f"  outer fold {outer_number}: "
            f"base={rmse(y[outer_validation], nested_base[outer_validation]):.4f} "
            f"expert={rmse(y[outer_validation], nested_expert[outer_validation]):.4f}",
            flush=True,
        )

    base_fold_scores = [
        rmse(y[validation], nested_base[validation])
        for _, validation in outer_folds
    ]
    expert_fold_scores = [
        rmse(y[validation], nested_expert[validation])
        for _, validation in outer_folds
    ]
    normal = y < 50
    severe = ~normal
    summary = {
        "validation": "strict nested GroupKFold: 5 outer x 3 inner",
        "outer_fold_seed": fold_seed,
        "parameters_fixed_before_validation": True,
        "base_nested_rmse": rmse(y, nested_base),
        "expert_nested_rmse": rmse(y, nested_expert),
        "gain": rmse(y, nested_base) - rmse(y, nested_expert),
        "base_fold_rmses": base_fold_scores,
        "expert_fold_rmses": expert_fold_scores,
        "normal_lt_50": {
            "base": rmse(y[normal], nested_base[normal]),
            "expert": rmse(y[normal], nested_expert[normal]),
        },
        "severe_ge_50": {
            "base": rmse(y[severe], nested_base[severe]),
            "expert": rmse(y[severe], nested_expert[severe]),
        },
    }

    suffix = "" if fold_seed is None else f"_seed{fold_seed}"
    np.save(RESULTS_DIR / f"nested_base_oof{suffix}.npy", nested_base)
    np.save(RESULTS_DIR / f"nested_tail_expert_oof{suffix}.npy", nested_expert)
    (RESULTS_DIR / f"nested_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n=== STRICT NESTED RESULT ===")
    print(f"Outer fold seed    : {fold_seed}")
    print(f"Base nested RMSE   : {summary['base_nested_rmse']:.4f}")
    print(f"Expert nested RMSE : {summary['expert_nested_rmse']:.4f}")
    print(f"Gain               : {summary['gain']:+.4f}")
    print(f"Base folds         : {[round(value, 4) for value in base_fold_scores]}")
    print(f"Expert folds       : {[round(value, 4) for value in expert_fold_scores]}")
    print(
        f"Normal <50         : {summary['normal_lt_50']['base']:.4f} -> "
        f"{summary['normal_lt_50']['expert']:.4f}"
    )
    print(
        f"Severe >=50        : {summary['severe_ge_50']['base']:.4f} -> "
        f"{summary['severe_ge_50']['expert']:.4f}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help="Shuffle patient group labels before the outer GroupKFold.",
    )
    arguments = parser.parse_args()
    run_nested_validation(arguments.fold_seed)
