# Simulation Study

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

One of the 200-state fits produces this report:

```text
================================================================================
                   Dynamic Discrete Choice Estimation Results
================================================================================
Method:      NFXP (Nested Fixed Point)    Observations:  7,500
Optimizer:   BHHH                         Individuals:   300
Family:      structural (linear utility)  Discount (beta): 0.95
                                          Scale (sigma):  1.0

[1] DATA
  State space:          200 states x 2 actions
  Periods per individual: ~25
  State coverage:       186/200 visited (93%)
  Single-action states: 76

[2] PRE-ESTIMATION CHECKS
  Reward features (K):  3
  Design rank:          3/3         Condition:          1.0e+00
  Contrast rank:        3/3       Contrast condition: 1.0e+00

[3] TRANSITION MODEL
  Transition source: supplied action-specific tensor

[4] RESULTS
  4a. Estimation
                          coef   std err       t   P>|t|   [0.025   0.975]
    theta_0             1.1238    0.0633   17.76   0.000   0.9997   1.2478
    theta_1            -0.7481    0.0429  -17.42   0.000  -0.8322  -0.6639
    theta_2             0.5976    0.0460   12.98   0.000   0.5074   0.6878
  4b. Identification
    Hessian condition:  4.1     Min eigenvalue: 199.63
  4c. Inference and fit
    Converged:      yes
    Iterations:     5
    SE method:      robust sandwich
    Log likelihood: -4,470.69
    Accuracy:       68.3%
================================================================================
```

## Inference

The inference experiment uses 1,000 independent panels. Each panel has 10,000
choices over 40 states. The empirical standard deviation measures how much the
estimate actually varies across panels. The mean standard error is what NFXP
reports from a single panel.

| Parameter | Truth | Mean | Actual SD | Mean SE | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reward feature 1 | 0.35 | 0.349 | 0.0353 | 0.0351 | 95.4% |
| Reward feature 2 | -0.25 | -0.252 | 0.0480 | 0.0482 | 94.8% |
| Reward feature 3 | 0.20 | 0.197 | 0.0522 | 0.0527 | 95.4% |

The reported standard errors closely match the observed variation across
panels. Coverage remains near 95 percent. Misses below the interval range from
1.6 to 2.8 percent, while misses above it range from 2.4 to 3.0 percent.

## Counterfactuals

The fitted model is solved again after changing either the reward or the engine
deterioration process.

| Change | True shift | Policy error | Value loss |
| --- | ---: | ---: | ---: |
| Increase the first reward parameter by 1.0 | 0.0829 | 0.0064 | 0.0030 |
| Slow engine deterioration | 0.0454 | 0.0067 | 0.0018 |

The first numeric column shows how much behavior truly changes after the
intervention. The second shows the distance between the fitted counterfactual
policy and the true-parameter counterfactual policy. Expected-value loss
measures the cost of using the fitted policy instead of the true-parameter
policy.

## Reproduce the Study

Run the study from the repository root:

```bash
PYTHONPATH=src:. uv run python validation/estimators/nfxp/ready.py
```

The [simulation code](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nfxp/ready.py)
and [reported results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp_ready.json)
contain the full experiment configuration.

## Related Studies

NFXP is compared against the full structural and IRL rosters in six simulation
studies: [bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
