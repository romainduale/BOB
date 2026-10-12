"""
tests/test_submission_format.py — Minimal pytest for Kaggle submission CSV format.

Checks:
  - Header is exactly "Index,target"
  - Exactly 11 013 rows (one per test visit)
  - Index matches X_test.csv index exactly
  - No NaN values in the target column
  - Target values are in [0, 132] (valid MDS-UPDRS range)

Usage:
    python -m pytest tests/test_submission_format.py -v
    python -m pytest tests/test_submission_format.py -v -k "11_final"
"""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
X_TEST_PATH     = PROJECT_ROOT / "data" / "X_test.csv"

# Expected constants
EXPECTED_ROWS = 11_013
MDS_UPDRS_MIN = 0
MDS_UPDRS_MAX = 132


def get_submission_files():
    """Return all submission CSV paths."""
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(SUBMISSIONS_DIR.glob("*.csv"))


def submission_ids():
    return [p.name for p in get_submission_files()]


@pytest.fixture(scope="module")
def x_test_index():
    """Load X_test index once per module."""
    return pd.read_csv(X_TEST_PATH, index_col="Index").index


@pytest.mark.parametrize("csv_path", get_submission_files(), ids=submission_ids())
def test_submission_header(csv_path):
    """Header must be exactly 'Index,target'."""
    with open(csv_path) as f:
        header = f.readline().strip()
    assert header == "Index,target", (
        f"{csv_path.name}: expected header 'Index,target', got '{header}'"
    )


@pytest.mark.parametrize("csv_path", get_submission_files(), ids=submission_ids())
def test_submission_row_count(csv_path):
    """Must have exactly 11 013 data rows."""
    df = pd.read_csv(csv_path)
    assert len(df) == EXPECTED_ROWS, (
        f"{csv_path.name}: expected {EXPECTED_ROWS} rows, got {len(df)}"
    )


@pytest.mark.parametrize("csv_path", get_submission_files(), ids=submission_ids())
def test_submission_no_nan(csv_path):
    """No NaN allowed in any column."""
    df = pd.read_csv(csv_path)
    nan_count = df.isna().sum().sum()
    assert nan_count == 0, (
        f"{csv_path.name}: {nan_count} NaN values found"
    )


@pytest.mark.parametrize("csv_path", get_submission_files(), ids=submission_ids())
def test_submission_index_matches_x_test(csv_path, x_test_index):
    """Index column must match X_test.csv index exactly (same values, same order)."""
    df = pd.read_csv(csv_path)
    assert list(df["Index"]) == list(x_test_index), (
        f"{csv_path.name}: 'Index' column does not match X_test.csv index"
    )


@pytest.mark.parametrize("csv_path", get_submission_files(), ids=submission_ids())
def test_submission_target_in_valid_range(csv_path):
    """Target values must be in [0, 132] (MDS-UPDRS motor score range)."""
    df = pd.read_csv(csv_path)
    out_of_range = ((df["target"] < MDS_UPDRS_MIN) | (df["target"] > MDS_UPDRS_MAX)).sum()
    assert out_of_range == 0, (
        f"{csv_path.name}: {out_of_range} predictions outside [{MDS_UPDRS_MIN}, {MDS_UPDRS_MAX}]"
    )
