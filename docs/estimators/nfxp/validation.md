# Simulation Study

## Important Links

- [NFXP Overview](../nfxp.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Counterfactuals](counterfactuals.md)
- [Bus Engine Example](rust_bus.md)
- [Simulation Studies](../../simulation_studies/index.md)

The study separates estimation, inference, and counterfactual analysis. Each
experiment simulates a new panel from a model whose reward and transition
process are fully specified.

## Estimation

The larger problem has 200 states and two actions. Keeping the engine allows it
to deteriorate. Replacing it returns it to the initial state. The reward has
three parameters and the model is estimated from 7,500 choices in each of 20
independent panels.

| Parameter | True value | Median relative error | 90th percentile relative error |
| --- | ---: | ---: | ---: |
| Reward feature 1 | 1.00 | 0.065 | 0.111 |
| Reward feature 2 | -0.80 | 0.032 | 0.065 |
| Reward feature 3 | 0.60 | 0.033 | 0.082 |

All 20 panels produced an estimate. The mean policy distance was 0.0085. This
distance is the average, across states, of the total variation between the two
action probabilities. Zero means the estimated and true policies agree in
every state.

The result file includes the report from one 200-state fit.

## Inference

The inference experiment uses 1,000 independent panels. Each panel has 10,000
choices over 40 states. The empirical standard deviation measures how much the
estimates vary across panels. The mean standard error averages the uncertainty
NFXP reports for the individual panels.

| Parameter | Truth | Mean | Actual SD | Mean SE | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reward feature 1 | 0.35 | 0.349 | 0.0353 | 0.0351 | 95.4% |
| Reward feature 2 | -0.25 | -0.252 | 0.0480 | 0.0482 | 94.8% |
| Reward feature 3 | 0.20 | 0.197 | 0.0522 | 0.0527 | 95.4% |

The reported standard errors closely match the observed variation across
panels. Coverage remains near 95 percent. Misses below the interval range from
1.6 to 2.8 percent, while misses above it range from 2.4 to 3.0 percent.

## Bootstrap Intervals

A separate experiment uses 50 independent panels. Each interval comes from 99
pairs-cluster resamples of complete individual trajectories.

| Parameter | Coverage | Mean interval width |
| --- | ---: | ---: |
| Reward feature 1 | 94.0% | 0.1324 |
| Reward feature 2 | 96.0% | 0.1809 |
| Reward feature 3 | 96.0% | 0.1955 |

All 50 panels returned an estimate. Every panel completed all 99 resamples, and
no bootstrap draw failed. Repeating the same 19-resample check produced the
same estimates, standard errors, intervals, coverage indicators, and failure
records.

## Prediction on New Panels

Each 200-state fit was evaluated on a new panel with 2,500 choices. The oracle
uses the true reward parameters on those same choices.

| Score | Fitted model | Excess over oracle |
| --- | ---: | ---: |
| Negative log likelihood | 0.5942 | 0.00038 |
| Brier score | 0.4085 | 0.00031 |

The excess is the fitted score minus the oracle score. Smaller values indicate
that estimated choice probabilities remain close to the true probabilities on
new trajectories.

## Counterfactuals

The fitted model is solved again after changing either the reward or the engine
deterioration process.

| Change | True shift | Policy error | Value loss |
| --- | ---: | ---: | ---: |
| Increase the first reward parameter by 1.0 | 0.0829 | 0.0064 | 0.0030 |
| Slow engine deterioration | 0.0454 | 0.0067 | 0.0018 |

`True shift` shows how much behavior truly changes after the intervention.
`Policy error` is the distance between the fitted counterfactual policy and the
true-parameter counterfactual policy. `Value loss` measures the cost of using
the fitted policy instead of the true-parameter policy.

## Reproduce the Study

Run the study from the repository root:

```bash
PYTHONPATH=src:. uv run python validation/estimators/nfxp/ready.py \
  --quiet --output validation/results/nfxp_ready.json

PYTHONPATH=src:. uv run python \
  validation/estimators/nfxp/bootstrap_calibration.py \
  --quiet --output validation/results/nfxp_bootstrap_calibration.json
```

**Result**

```text
wrote validation/results/nfxp_ready.json
wrote validation/results/nfxp_bootstrap_calibration.json
```

The [simulation code](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nfxp/ready.py)
and [reported results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp_ready.json)
contain the full experiment configuration. The bootstrap
[code](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nfxp/bootstrap_calibration.py)
and [results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp_bootstrap_calibration.json)
contain the resampling design and interval results.

## Related Studies

NFXP is compared with the package's structural and IRL estimators in six
simulation studies: [bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
