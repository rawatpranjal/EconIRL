# Wulfmeier-Shaped Study

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Simulation Study](validation.md)
- [Quick Start](quick_start.md)
- [Generated result file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl_wulfmeier.json)

Wulfmeier, Ondruska, and Posner (2015) study nonlinear rewards in grid
environments. The generated comparison here follows that problem shape.

## Problem

Objectworld rewards depend on distances to colored objects. Binaryworld
rewards depend on local binary features. Neural and linear MCE-IRL receive the
same task features. The neural model can combine them nonlinearly. Both
environments use a 32 by 32 grid, five actions, 30 percent random
demonstration actions, and a discount factor of 0.9.

## Study Design

The study varies the number of demonstrations over 8, 16, 32, 64, and 128.
Five generated panels are fitted at each sample size. Neural MCE-IRL uses three
training seeds per panel. Linear MCE-IRL is fitted once to each panel.

Each neural fit uses a 200-epoch budget. Because 30 percent of demonstration
actions are random, exact occupancy matching is not expected. The result file
reports convergence and termination counts. A linear fit enters the comparison
only when optimization succeeds and its occupancy and Bellman residuals are
within the reported tolerances.

The controlled [Simulation Study](validation.md) provides the estimator
convergence evidence.

Expected value difference compares the true-reward value of the reference
policy with the true-reward value of each learned policy. Both values exclude
the entropy bonus. Lower values are better. An independently generated map
measures transfer through the same feature representation.

## Results

![Generated reward maps and expected value difference curves](../../_static/estimators/deep_mce_irl_wulfmeier.png)

The reward panels show the true reward and the neural fit with median expected
value difference among the 15 fits at 128 demonstrations. Neural curves pool
five generated panels and three training seeds. Linear curves use the five
generated panels.

All 200 requested fits produced finite results. Of these, 143 met the stated
stopping tolerances. The linked result file reports aggregate convergence and
termination counts. The JSONL checkpoint retains one record for each fit. The
result file records nine criteria covering completeness, finite results,
optimizer solutions, and the reported neural-linear comparisons. All nine were
satisfied.

| Environment and demonstrations | Metric | Neural | Linear |
| --- | --- | ---: | ---: |
| Objectworld, 128 | EVD | 0.224 | 0.556 |
| Objectworld, 128 | Transfer EVD | 0.123 | 0.439 |
| Binaryworld, 64 | EVD | 1.847 | 5.345 |
| Binaryworld, 64 | Transfer EVD | 2.638 | 5.402 |
| Binaryworld, 128 | EVD | 1.684 | 5.352 |
| Binaryworld, 128 | Transfer EVD | 2.403 | 5.396 |

The table reports the comparison cells selected before fitting. Each linear
entry summarizes five fits that completed optimization and met the occupancy
and Bellman tolerances. Each neural entry summarizes 15 fits.

## Scope

The study is a generated comparison. It is not a replication of published
numbers. The paper reports its main comparisons in figures rather than a
recoverable numerical table.

## Reproduce the Study

The runnable program is
[`wulfmeier.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/wulfmeier.py).
It checkpoints each fit and records failures instead of replacing them with
successful values.

```bash
PYTHONPATH=src:. python validation/estimators/deep_mce_irl/wulfmeier.py \
  --checkpoint validation/results/deep_mce_irl_wulfmeier.jsonl \
  --output validation/results/deep_mce_irl_wulfmeier.json \
  --quiet
```

**Result**

```text
wrote validation/results/deep_mce_irl_wulfmeier.json
status: ready
```
