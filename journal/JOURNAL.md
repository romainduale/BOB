# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's MDS-UPDRS score prediction
- **Goal:** Predict MDS-UPDRS motor scores (continuous regression target) for Parkinson's disease patients
- **Last experiment:** 01_dummy - done
- **Last result:** DummyRegressor(mean), 80/20 holdout — pushed to hub as `01_dummy`

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2024-10-07
  - env manager: pip+venv - recorded: 2024-10-07
  - agent feature: installed - recorded: 2024-10-07
  - optional features: none - recorded: 2024-10-07
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: <local | hub | mlflow> - recorded: <YYYY-MM-DD>
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: <mlflow-tracking-uri | n/a> - recorded: <YYYY-MM-DD>
  - student prior: <beginner | some-sklearn | comfortable> - recorded: <YYYY-MM-DD>
  - CV splitter family: <KFold | StratifiedKFold | GroupKFold | TimeSeriesSplit | other> - recorded: <YYYY-MM-DD>

## Data understanding (EDA)

- **Status:** done - 2024-10-07
- **Summary:** 44,590 train rows × 13 cols (+ 11,013 test rows); continuous MDS-UPDRS regression target (mean ≈ 37, SD ≈ 16.5). `patient_id` repeats (~8 rows/patient) → `GroupKFold` on patient. Heavy missingness in `time_since_intake_off` (79 %), `on`/`off` (30–42 %), `ledd` (37 %). `off` correlates with target at r = 0.87 — **potential leakage**, requires clarification before the baseline.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) — lower-bound baseline | done | See hub: [01_dummy](https://skore.probabl.ai/bob/01_dummy/estimators/57306) | — |

## Backlog

| # | Item | Source |
|---|---|---|
