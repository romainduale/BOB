# Experiment 08 — Regularized residual expert

## Goal

Improve the Exp-07 winner on large target values without degrading ordinary
predictions. The experiment adds one small, strongly regularized
`HistGradientBoostingRegressor` that predicts the residual of Exp-07. Its
features are available at inference time: the base prediction, raw covariates,
missing-value indicators, and target-free per-patient aggregates.

No Kaggle target, test target, or validation target is used as a feature. Every
reported validation split is grouped by `patient_id`.

## Fixed model

- 80 boosting iterations
- learning rate 0.04
- maximum depth 2
- minimum 80 samples per leaf
- L2 regularization 10
- final predictions clipped to the valid `[0, 132]` range

This configuration was fixed after a small screening. Weighted and gated tail
variants were rejected because their global OOF RMSE was worse.

## Comparable Exp-07 OOF evaluation

| Model | Global OOF RMSE |
| --- | ---: |
| Exp-07 clipped baseline | 3.9176 |
| Exp-08 residual expert | **3.8943** |

The fixed expert also scored 3.8909 on a second patient-grouped meta split.

| Target segment | Exp-07 | Exp-08 |
| --- | ---: | ---: |
| `< 50` | 3.3558 | **3.3527** |
| `>= 50` | 5.3269 | **5.2611** |
| `> 70` | 7.7664 | **7.5492** |

For targets above 70, mean underprediction decreased from about 5.26 to 4.86.

## Strict nested validation

The full base chain and residual expert were then rebuilt inside 5 outer × 3
inner patient-grouped folds. Hyperparameters remained fixed.

| Outer split | Nested base | Nested expert | Gain | Improved folds |
| --- | ---: | ---: | ---: | ---: |
| Primary | 3.8920 | **3.8522** | +0.0398 | 5/5 |
| Shuffled groups, seed 7 | 3.8947 | **3.8606** | +0.0341 | 5/5 |

Across both independent outer splits, the expert improved all 10 folds. In the
primary nested run, RMSE changed from 3.3351 to 3.3194 below 50 and from 5.2894
to 5.1982 at or above 50. In the seed-7 run, the corresponding changes were
3.3398 to 3.3253 and 5.2886 to 5.2123.

Nested scores are used to confirm the direction and stability of the gain. They
are not directly interchangeable with the original Exp-07 OOF number because
the base components are refitted inside three inner folds.

## Reproduction

Run the lightweight final experiment:

```bash
./.venv/bin/python experiments/08_tail_expert.py
```

Run the two strict validations:

```bash
./.venv/bin/python experiments/08b_nested_validation.py
./.venv/bin/python experiments/08b_nested_validation.py --fold-seed 7
```

Generated results are stored under `results/exp08/` and intentionally ignored
by Git. The local candidate file is `submissions/08_tail_expert.csv`; it has
11,013 rows, the exact sample-submission index, no duplicates, and no missing
predictions. It was not submitted to Kaggle.
