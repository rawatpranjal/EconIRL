# Simulation Study

## Important Links

- [MCE-IRL overview](../mce_irl.md)
- [Quick start](quick_start.md)
- [Pre-estimation checks](pre_estimation.md)
- [Cross-estimator simulation studies](../../simulation_studies/index.md)

MCE-IRL has three complementary checks. A compact generated study measures
reward and counterfactual recovery. A repeated-run study checks inference. A
road-choice study checks the large deterministic task structure used by
Ziebart et al. (2008).

The numerical results are in
[`mce_irl_ready.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mce_irl_ready.json),
[`mce_irl_ziebart_synthetic.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mce_irl_ziebart_synthetic.json),
and
[`mce_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mce_irl.json).

## Repeated-Run Inference

The inference study uses a one-step model with a closed-form reward parameter.
It runs the public estimator on 300 independently generated panels. Each panel
has 400 trajectories.

| Check | Result |
| --- | ---: |
| Converged fits | 300 / 300 |
| True parameter | 0.4000 |
| Mean estimate | 0.4058 |
| Bias | 0.00584 |
| RMSE | 0.0986 |
| 95 percent interval coverage | 0.960 |
| Mean asymptotic SE | 0.1022 |
| Monte Carlo SD | 0.0985 |
| Asymptotic / bootstrap SE ratio for one panel | 1.057 |
| Largest stationarity residual | 6.62e-9 |

The standard-error comparison uses 200 bootstrap refits of one generated
400-trajectory panel. The repeated-run rows above use 300 independent panels.
A reward intervention changes the action probability by 0.205. All seven
checks pass.

```bash
PYTHONPATH=src:. uv run python validation/estimators/mce_irl/ready.py --quiet
```

**Result**

```text
wrote validation/results/mce_irl_ready.json
status: ready
```

## Ziebart Road Structure

The road study generates data. It does not use the Pittsburgh taxi traces or
their fitted road graph. The generated experiment follows the observable data
and model structure reported by Ziebart et al. (2008):

| Contract | Generated value |
| --- | ---: |
| Deterministic directed road-segment states | 302,500 |
| Intersection transition actions | 907,500 |
| Raw feature counts | 22 |
| Identified fit features | 19 |
| Feature families | road type, speed, lanes, turns |
| Raw trips | 13,220 |
| Generated drivers | 25 |
| Origin-destination tasks | 64 |
| Trips discarded as short, cyclic, or noisy | 3,966 |
| Training trips | 1,851 |
| Test trips | 7,403 |

Each origin-destination pair is an `MCEIRLTask`. Its destination is absorbing.
The candidate class has a common route prefix, a bounded branching region, and
a common suffix. Every legal path reaches the destination. The 64 tasks are
compact views of one spatial road graph and share one linear reward vector.

The adapter retains all 22 observed counts. It selects a numerically scaled,
full-rank 19-feature action-contrast basis for estimation. The public `MCEIRL`
estimator then fits 7,552 compiled task states. It contains no road-specific
estimation logic. The fit takes about 4.1 seconds and has a stationarity
residual of 7.37e-9. The complete run takes about 7.6 seconds.

```bash
PYTHONPATH=src:. uv run python validation/estimators/mce_irl/ziebart_road_synthetic.py --quiet
```

**Result**

```text
wrote validation/results/mce_irl_ziebart_synthetic.json
status: passed
```

The generated-data metrics and the published Table 1 values are:

| Metric | Generated data | Published MaxEnt paths |
| --- | ---: | ---: |
| Held-out distance match | 80.82 percent | 78.79 percent |
| Held-out routes with at least 90 percent match | 30.15 percent | 52.98 percent |
| Training-path average log probability | -3.86 | -6.85 |

Those values are literature targets. They are not acceptance thresholds for
generated data. The comparison checks metric definitions and exposes where the
generated route distribution differs. Reproducing Table 1 requires the original
road graph, fitted routes, candidate path sets, and split.

## Generated Recovery

The primary compact study uses 25 states, 3 actions, and 8 action-dependent
reward features. It checks reward, policy, value, Q, and three counterfactual
families.

| Metric | Result |
| --- | ---: |
| Feature residual | 1.89e-12 |
| Occupancy residual | 0.00106 |
| Reward normalized RMSE | 0.0823 |
| Policy total variation | 0.00698 |
| Type A regret | 0.000433 |
| Type B regret | 0.000410 |
| Type C regret | 0.000094 |

MCE-IRL also appears in the public bus engine, taxi gridworld, route choice,
stockpiling, fleet maintenance, and vehicle scrappage studies.
