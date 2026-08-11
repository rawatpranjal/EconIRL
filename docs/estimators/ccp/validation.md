# Simulation Study

## Important Links

- [CCP Overview](../ccp.md)
- [Quick Start](quick_start.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Counterfactuals](counterfactuals.md)
- [Simulation Studies](../../simulation_studies/index.md)

The study separates estimation, inference, prediction on new panels, and
counterfactual analysis. Each experiment simulates a new panel from a specified
reward and transition process. Every fit receives the data-generating
transition tensor, so the results do not include uncertainty from transition
estimation.

## Estimation

The estimation problem has 100 states and two actions. The reward has three
parameters. Three-stage NPL is estimated from 160,000 choices in each of 20
independent panels.

| Parameter | True value | Median relative error | 90th percentile relative error |
| --- | ---: | ---: | ---: |
| Reward feature 1 | 0.35 | 0.029 | 0.063 |
| Reward feature 2 | -0.25 | 0.019 | 0.064 |
| Reward feature 3 | 0.20 | 0.052 | 0.095 |

All 20 panels produced an estimate. The mean policy distance was 0.0028. The
largest fit took 8.8 seconds. Policy distance is the average total variation
between the fitted and true action probabilities across states.

## Inference

The inference experiment uses 1,000 independent panels. Each panel has 10,000
choices over 20 states. The empirical standard deviation measures variation
across panels. The mean standard error averages the robust uncertainty estimate
reported for each panel.
These robust standard errors come from the fixed-CCP pseudo-likelihood. They
treat the empirical CCPs and supplied transition tensor as fixed. They do not
propagate uncertainty from estimating CCPs or transitions.

| Parameter | True value | Mean estimate | Empirical SD | Mean SE | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reward feature 1 | 0.35 | 0.354 | 0.0305 | 0.0296 | 94.2% |
| Reward feature 2 | -0.25 | -0.248 | 0.0506 | 0.0499 | 95.4% |
| Reward feature 3 | 0.20 | 0.203 | 0.0521 | 0.0543 | 96.3% |

Mean standard errors are 0.97 to 1.04 times the empirical standard deviations.
Lower-tail miss rates range from 2.4 to 3.6 percent. Upper-tail miss rates
range from 1.2 to 2.2 percent.

All four standard-error methods were applied to one 40-state panel:

| Method | Feature 1 SE | Feature 2 SE | Feature 3 SE |
| --- | ---: | ---: | ---: |
| Asymptotic | 0.0386 | 0.0522 | 0.0536 |
| Robust | 0.0387 | 0.0523 | 0.0533 |
| Clustered | 0.0379 | 0.0532 | 0.0540 |
| Pairs-cluster bootstrap | 0.0375 | 0.0508 | 0.0546 |

The clustered estimates are between 0.99 and 1.05 times their bootstrap
counterparts in this panel. This single-panel comparison does not measure
repeated-sample coverage or interval width. A separate calibration uses 50
independent panels. Each interval uses 99 pairs-cluster resamples of complete
individual trajectories. The bootstrap re-estimates empirical CCPs and keeps
the supplied transition tensor fixed. This comparison does not test the full
Hotz-Miller two-step covariance or the Kasahara-Shimotsu parametric bootstrap.

All 50 calibration panels completed, with no failed resamples. Coverage was
0.98, 0.98, and 0.90 across the three reward parameters. Mean interval widths
were 0.1130, 0.1874, and 0.2103. Repeating the same 19-draw bootstrap program
twice produced identical estimates, intervals, and failure records.

## Prediction on New Panels

Each 100-state fit was evaluated on a new panel with 2,500 choices. Across the
20 panels, mean negative log likelihood was 0.682214 and mean Brier score was
0.489172. The mean excess over oracle was 0.000039 for negative log likelihood
and 0.000037 for Brier score.

## Counterfactuals

The fitted model is solved again after changing either the first reward
parameter or the deterioration process.

| Change | Mean policy TV | Mean value loss |
| --- | ---: | ---: |
| Increase the first reward parameter by 1.0 | 0.0022 | 0.000184 |
| Slow deterioration | 0.0019 | 0.000083 |

Policy TV is the state-averaged total-variation distance between the fitted
counterfactual policy and the policy from the true parameters. Value loss
measures the cost of using the fitted policy instead of the true-parameter
policy.

## Reproduce the Study

Run the study from the repository root:

```bash
PYTHONPATH=src:. uv run python validation/estimators/ccp/ready.py \
  --quiet --output validation/results/ccp_ready.json

PYTHONPATH=src:. uv run python \
  validation/estimators/ccp/bootstrap_calibration.py \
  --quiet --output validation/results/ccp_bootstrap_calibration.json
```

**Result**

```text
wrote validation/results/ccp_ready.json
status: ready
wrote validation/results/ccp_bootstrap_calibration.json
```

The [simulation code](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ccp/ready.py)
and [reported results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp_ready.json)
contain the full experiment configuration. The
[bootstrap code](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ccp/bootstrap_calibration.py)
and [bootstrap results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp_bootstrap_calibration.json)
contain the repeated calibration.

## Related Studies

CCP is compared with the package's structural and IRL estimators in six
simulation studies: [bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
