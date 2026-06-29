# Simulation Study

GLADIUS is evaluated on two related high-dimensional-state synthetic cells.
Each cell has 21 discrete states paired with a high-dimensional encoded
state-feature vector, a low-dimensional action-dependent reward basis, and an
anchor action whose rewards are known. All recovery claims are compared against
the truth from the data-generating process.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/gladius/run.py).
It writes two results files:
[`gladius.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json)
(primary cell) and
[`gladius_scaled.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_scaled.json)
(scaled cell).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/gladius/run.py
```

The primary and scaled cells share the same structural support: full feature
rank, full state-action coverage, and a valid anchor. The raw Bellman reward
and value NRMSE fail their checks on both cells. Projected reward, policy, and
Q recover reasonably. The absolute reward level implied by the raw Bellman
equation does not. Counterfactual regret is small, but low regret alone is not
sufficient for structural validity when the upstream reward checks fail.

## Evidence

GLADIUS is compared against the full structural and IRL rosters on the
[bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md), and
[direct optimization](../../simulation_studies/direct_optimization.md) pages.
GLADIUS also appears on the
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md), and
[fleet maintenance](../../simulation_studies/fleet_maintenance.md) pages. See
the [simulation studies index](../../simulation_studies/index.md) for what
each study shows.
