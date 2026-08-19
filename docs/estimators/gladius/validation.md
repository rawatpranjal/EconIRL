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

The complete report also requires installed-wheel notebook execution and exact
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

### Result

The full design ran at 800 epochs with no update cap. Five of the six cells sit
inside the paper bound. The largest cell does not.

| Trajectories | EconIRL MAPE (%) | Paper mean (SE) | Bound | Inside |
| --- | --- | --- | --- | --- |
| 50 | 0.97 | 3.44 (1.28) | 6.00 | yes |
| 250 | 0.47 | 0.84 (0.51) | 1.86 | yes |
| 500 | 0.33 | 0.55 (0.20) | 0.95 | yes |
| 1000 | 0.29 | 0.52 (0.22) | 0.96 | yes |
| 2500 | 0.22 | 0.13 (0.06) | 0.25 | yes |
| 5000 | 0.26 | 0.12 (0.06) | 0.24 | no |

So GLADIUS is not a completed Table 2 replication. Treat the paper column as
matched for `N` up to 2500 and unmatched at `N=5000`.

Two facts qualify the miss. The NFXP oracle control in the same harness beats
the paper's Rust column at every sample size, so the harness is not producing
the gap. And the 20 seeds at `N=5000` span 0.130 to 0.501, so the best seeds
already reach the paper value and the miss is variance in the tail.

A second full run under the paper author's batch size, with a floor of ten Q
updates per epoch, is stored beside the receipt as
[`gladius_paper_table2_author_batch_diagnostic.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_paper_table2_author_batch_diagnostic.json).
It brings `N=5000` to 0.16, confirmed at 0.15 on eight further seeds, and pushes
`N=2500` out to 0.32. Neither batch rule clears all six cells. The largest-cell
result is therefore sensitive to optimizer batching rather than fixed by the
method, which is an open lead. The reported receipt is the first design, chosen
before any result was seen, because selecting the batch rule that clears the
acceptance statistic would not be evidence.

## Bootstrap Calibration

The prespecified design uses 20 independently simulated panels and 19 whole-
trajectory bootstrap draws per panel. It checks three reward and three policy
functionals per panel using point-centered bootstrap-standard-error intervals.
The controlled nominal states use a categorical encoder and the estimator seed
is held fixed while panel and trajectory-resampling seeds vary. The requirements
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
scientific check fails, serialization did not import the installed wheel, or a
notebook code cell was not executed cleanly.

