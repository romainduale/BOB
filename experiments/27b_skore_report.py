"""Publish the Exp-27 patient-disjoint holdout as a Skore EstimatorReport.

The report key intentionally matches the Kaggle CSV stem.  This script aborts
instead of overwriting an existing key and never uploads a Kaggle submission.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp27"
REPORT_KEY = "27_public_clean_emg_ensemble"
PROJECT_NAME = "bobathon-esilv"

REFERENCE_URL = "https://github.com/Harry01530/bobathon-esilv.git"
REFERENCE_COMMIT = "a608a6f86a0f7886f46982bf19535fcfb8b99cd6"
REFERENCE_ROOT = Path("/private/tmp") / f"harry-bobathon-{REFERENCE_COMMIT[:12]}"


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
    actual = subprocess.run(
        ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != REFERENCE_COMMIT:
        raise RuntimeError(f"Reference commit mismatch: {actual}")
    return REFERENCE_ROOT


def main() -> None:
    reference_root = ensure_reference_checkout()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(reference_root / "src"))

    from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
    from parkinsons.hub import load_skore_credentials
    from parkinsons.two_stage_mle import TwoStageMLE

    reference_data = reference_root / "data" / "Kaggle_officiel"
    train = pd.read_csv(reference_data / "X_train.csv")
    test = pd.read_csv(reference_data / "X_test.csv")
    y_frame = pd.read_csv(DATA_DIR / "y_train.csv").set_index("Index")
    y = y_frame.loc[train["Index"], "target"].to_numpy(dtype=float)
    groups = train["patient_id"].to_numpy()
    fit_indices, validation_indices = list(
        GroupKFold(n_splits=5).split(train, y, groups=groups)
    )[4]
    if set(groups[fit_indices]) & set(groups[validation_indices]):
        raise RuntimeError("Patient leakage in report holdout")

    fit_frame = train.iloc[fit_indices].reset_index(drop=True)
    validation_frame = train.iloc[validation_indices].reset_index(drop=True)
    category_reference = pd.concat([validation_frame, test], ignore_index=True)
    cat_dtypes = make_cat_dtypes(fit_frame, category_reference)
    estimator = TwoStageMLE(
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

    report = skore.EstimatorReport(
        estimator,
        X_train=fit_frame,
        y_train=y[fit_indices],
        X_test=validation_frame,
        y_test=y[validation_indices],
    )
    prediction = report.estimator_.predict(validation_frame)
    holdout_rmse = float(
        np.sqrt(mean_squared_error(y[validation_indices], prediction))
    )
    if not np.isfinite(holdout_rmse) or holdout_rmse > 2.75:
        raise RuntimeError(f"Unexpected holdout RMSE: {holdout_rmse:.6f}")

    cfg = load_skore_credentials()
    skore.login(mode="hub")
    project = skore.Project(
        name=PROJECT_NAME,
        mode="hub",
        workspace=cfg["workspace"],
    )
    existing = set(project.summarize().frame()["key"].astype(str))
    if REPORT_KEY in existing:
        raise RuntimeError(f"Skore key already exists: {REPORT_KEY}")

    skore.console.begin_capture()
    project.put(REPORT_KEY, report)
    captured = skore.console.end_capture()
    urls = re.findall(r"https://[^\s\]>]+", captured)
    estimator_urls = [url.rstrip("/") for url in urls if "/estimators/" in url]
    if not estimator_urls:
        raise RuntimeError(f"EstimatorReport URL not captured:\n{captured}")
    report_url = estimator_urls[-1]

    metadata = {
        "report_key": REPORT_KEY,
        "report_type": "EstimatorReport",
        "report_url": report_url,
        "holdout": "primary GroupKFold fold 5, patient-disjoint",
        "holdout_rmse": holdout_rmse,
        "n_train": len(fit_indices),
        "n_test": len(validation_indices),
        "reference_commit": REFERENCE_COMMIT,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "skore_report.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
