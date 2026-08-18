# Simulation Study

GLADIUS has three separate scientific evidence surfaces. They answer different
questions and should not be collapsed into one result.

1. The paper Table 2 replication checks the shared-trunk minimax training
   recipe on the authors' bus experiment.
2. The oracle-simulation structural stress test checks reward, Q, value, policy, and
   re-solved counterfactuals against oracle objects.
3. The trajectory-bootstrap calibration checks reward and policy interval
   coverage, tail balance, draw success, width stability, and exact seeded
   reproducibility.

The complete gate also requires installed-wheel notebook execution and exact
fresh-process serialization parity.

## Oracle-Simulation Structural Cell

The primary 21-state, 3-action cell passes all 12 prespecified checks:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Final outer-Q loss | 1.0927 | at most 2.0 |
| Parameter cosine | 0.9773 | at least 0.90 |
| Parameter relative RMSE | 0.3119 | at most 0.50 |
| Raw Bellman reward NRMSE | 0.2989 | at most 0.30 |
| Projected reward NRMSE | 0.1311 | at most 0.30 |
| Policy TV | 0.0187 | at most 0.12 |
| Value NRMSE | 0.2194 | at most 0.30 |
| Q NRMSE | 0.1237 | at most 0.30 |
| Type A regret | 0.00291 | at most 0.12 |
| Type B regret | 0.00814 | at most 0.12 |
| Type C regret | 0.00102 | at most 0.12 |

The cell uses the lower-level `anchor_moment` diagnostic with global anchor
level calibration. It evaluates the structural objects against fixed thresholds;
it is not presented as the paper's minimax replication. The separate Table 2
receipt validates that implementation path.

Run the cell with:

```bash
PYTHONPATH=src:. uv run python validation/estimators/gladius/run.py \
  --quiet-progress --enforce-gates
```

The tracked result is
[`gladius.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json).

## Paper Table 2

The exact paper-bus driver uses the 20 mileage states, `+1` through `+4`
maintenance transitions, replacement reset, true reward parameters `(1, 5)`,
and discount `0.95`. Qualification requires all six sample sizes and 20 seeds
per size. Each mean reward MAPE must be no larger than the reported paper mean
plus two reported standard errors, and cells after `N=250` may not deteriorate
relative to the `N=250` mean.

The checked-in author experiment selects the best epoch using true held-out
reward MAPE. The replication matches and labels that simulation-only oracle
selection. The public estimator never receives true rewards or held-out truth.

The current 128,000-update qualification attempt is retained as a failed
10-seed diagnostic, not promoted as a replication. Its large-sample cells
receive only 103, 41, and 21 epochs at `N=1000`, `N=2500`, and `N=5000`, and
already exceed the full-design error budget. Completing the remaining seeds
cannot make the prespecified mean-MAPE checks pass. GLADIUS therefore remains blocked
on this paper gate pending a reviewed paper-scale budget or an approved replica
protocol.

## Bootstrap Calibration

The prespecified design uses 20 independently simulated panels and 19 whole-
trajectory bootstrap draws per panel. It checks three reward and three policy
functionals per panel using point-centered bootstrap-standard-error intervals.
The controlled nominal states use a categorical encoder and the estimator seed
is held fixed while panel and trajectory-resampling seeds vary. Required gates
are at least 85% empirical coverage, no more than 10% misses in either tail, at
least 95% usable panels and successful draws, positive interval widths,
p95-to-median width no larger than 4, and exact seeded reproducibility. The
qualified receipt achieves reward coverage 0.967 and policy coverage 0.950.

## Combined Gate

After generating all receipts and executing the notebook from the built wheel:

```bash
PYTHONPATH=src:. uv run python validation/estimators/gladius/qualification_report.py
```

The report fails closed when a receipt is missing, a prespecified design changes, a
scientific gate fails, serialization did not import the installed wheel, or a
notebook code cell was not executed cleanly.

The exact commands and clean-HEAD rule are in the
[qualification runbook](qualification_runbook.md).
