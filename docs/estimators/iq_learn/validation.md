# Simulation Study

Read this page as an imitation and diagnostic validation page. Passing regret
checks is not enough here. Reward, value, Q, and coverage decide whether the
objects can be read structurally.

IQ-Learn runs on three synthetic cells covering low-dimensional tabular,
high-dimensional neural, and state-only reward settings. Each cell has known
transitions, policy, value, Q function, and counterfactual oracle objects, so
every recovery claim is checked against the truth. The primary cell is
`canonical_low_action`.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/iq_learn/run.py).
It writes the results file
[`iq_learn.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/iq_learn.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/iq_learn/run.py
```

The estimator passes imitation and counterfactual regret checks on the primary
cell. Reward, value, and Q recovery fail. Low regret on this cell reflects that
the Q-induced policy happens to produce near-oracle welfare under the applied
interventions, not that the reward or value objects are structurally accurate.
The high-dimensional neural cell and state-only cell fail most structural
recovery checks as well.

## Sparse-Support Guard

The sparse-support guard uses a tiny panel with one observed state and one
observed state-action pair (state coverage 0.333, state-action coverage
0.167). Even when all non-coverage metrics pass, the run is not
counterfactual-valid because support checks fail. The guard prevents future
changes from treating small policy or regret numbers as sufficient when the
expert panel does not cover the relevant state-action space.

```bash
PYTHONPATH=src:. python validation/estimators/iq_learn/sparse_support_guard.py
```

Results: [`iq_learn_sparse_support_guard.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/iq_learn_sparse_support_guard.json).

## Evidence

IQ-Learn is compared against the full structural and IRL rosters on the
[bus engine](../../simulation_studies/rust_bus.md) and
[taxi gridworld](../../simulation_studies/taxi_gridworld.md) pages. See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
