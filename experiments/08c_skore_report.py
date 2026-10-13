"""Publish an exact Skore audit report for the Exp-08 OOF predictions.

Exp-08 is a stacked model whose base layer is itself cross-fitted. Rebuilding
that full nested training graph inside ``skore.evaluate`` would produce a
different experiment. This report therefore exposes the already validated,
patient-grouped OOF predictions to Skore so its residual and diagnostic plots
match the reported 3.8943 RMSE exactly.

The estimator below is intentionally named ``OOFAuditRegressor``: it reads the
stored prediction column and does not pretend to be the deployable pipeline.
The reproducible training code lives in ``08_tail_expert.py`` and the stricter
fully retrained audit lives in ``08b_nested_validation.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp08"
PREDICTION_COLUMN = "exp08_oof_prediction"
PROJECT_NAME = "08-tail-expert"
REPORT_KEY = "08_tail_expert_oof_cv5"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class OOFAuditRegressor(RegressorMixin, BaseEstimator):
    """Expose precomputed leak-free OOF predictions to Skore diagnostics."""

    def __init__(self, prediction_column: str = PREDICTION_COLUMN):
        self.prediction_column = prediction_column

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OOFAuditRegressor":
        if self.prediction_column not in X.columns:
            raise ValueError(f"Missing prediction column: {self.prediction_column}")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        return X[self.prediction_column].to_numpy(dtype=float)


def report_identifier(project: skore.Project, key: str) -> str | None:
    """Return the newest persisted report id for ``key`` when available."""
    frame = project.summarize().frame()
    if frame.empty:
        return None
    reset = frame.reset_index()
    key_columns = [column for column in reset.columns if str(column).lower() == "key"]
    id_columns = [column for column in reset.columns if str(column).lower() == "id"]
    if not key_columns or not id_columns:
        return None
    matching = reset[reset[key_columns[0]] == key]
    if matching.empty:
        return None
    return str(matching.iloc[-1][id_columns[0]])


def main() -> None:
    X = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"].to_numpy()
    prediction = np.load(RESULTS_DIR / "oof_tail_expert.npy")

    if len(X) != len(y) or len(y) != len(prediction):
        raise ValueError("X, y, and Exp-08 OOF predictions have different lengths")
    if np.isnan(prediction).any():
        raise ValueError("Exp-08 OOF predictions contain NaN values")

    # Keep the Hub payload minimal: Skore receives only the target passed to
    # ``evaluate`` and this single OOF prediction column. Patient identifiers
    # and clinical covariates are used locally to build the folds, then omitted.
    report_X = pd.DataFrame(
        {PREDICTION_COLUMN: prediction},
        index=X.index,
    )
    folds = list(
        GroupKFold(n_splits=5).split(
            X, y, groups=X["patient_id"].to_numpy()
        )
    )
    for train_indices, validation_indices in folds:
        train_patients = set(X.iloc[train_indices]["patient_id"])
        validation_patients = set(X.iloc[validation_indices]["patient_id"])
        if train_patients.intersection(validation_patients):
            raise AssertionError("A patient appears in both train and validation")

    expected_folds = [
        rmse(y[validation_indices], prediction[validation_indices])
        for _, validation_indices in folds
    ]
    expected_global = rmse(y, prediction)
    print(f"Exp-08 global OOF RMSE: {expected_global:.4f}", flush=True)
    print(
        f"Expected grouped folds: {[round(value, 4) for value in expected_folds]}",
        flush=True,
    )

    report = skore.evaluate(
        OOFAuditRegressor(),
        report_X,
        y,
        splitter=iter(folds),
        n_jobs=1,
    )
    skore_folds = [float(fold_report.metrics.rmse()) for fold_report in report.reports_]
    if not np.allclose(skore_folds, expected_folds, atol=1e-10):
        raise AssertionError(
            f"Skore fold RMSE {skore_folds} does not match {expected_folds}"
        )

    config = json.loads((PROJECT_ROOT / ".skore").read_text())
    os.environ["SKORE_HUB_URI"] = config["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = config["api_key"]
    skore.login(mode="hub")

    project = skore.Project(
        PROJECT_NAME,
        mode="hub",
        workspace=config["workspace"],
    )
    project.put(REPORT_KEY, report)
    identifier = report_identifier(project, REPORT_KEY)

    print(f"Pushed Skore project: {PROJECT_NAME}", flush=True)
    print(f"Pushed report key: {REPORT_KEY}", flush=True)
    print(f"Skore fold RMSE: {[round(value, 4) for value in skore_folds]}", flush=True)
    print(f"Skore report id: {identifier or 'available in project summary'}", flush=True)
    print(
        f"Skore project URL: https://skore.probabl.ai/{config['workspace']}/"
        f"{PROJECT_NAME}/cross-validations/",
        flush=True,
    )


if __name__ == "__main__":
    main()
