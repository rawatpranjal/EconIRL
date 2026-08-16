# Simulation Study

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Wulfmeier-Shaped Study](wulfmeier_objectworld.md)
- [Counterfactuals](counterfactuals.md)
- [Result file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl_ready.json)
- [Bootstrap result](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl_bootstrap_calibration.json)

The controlled study uses a nonlinear state-action reward over 32 states and
three actions. Action 0 anchors the reward map. The transitions and true reward
are known to the simulator.

The study varies both the generated panel and neural initialization. It uses
100 panels and three training seeds per panel. These 300 fits measure
stability. They are not confidence intervals.

The fitted object is evaluated through anchored reward error, policy distance,
value error, Q-function error, and counterfactual regret. Raw network weights
are not compared.

## Stability Results

All 300 fits completed and satisfied the estimator's convergence criterion.
The results met all 18 study criteria.

| Metric | Median | 95th percentile |
| --- | ---: | ---: |
| Reward normalized RMSE | 0.0677 | 0.1198 |
| Policy total variation | 0.00850 | 0.01493 |
| Value normalized RMSE | 0.3056 | 1.8938 |
| Q normalized RMSE | 0.0689 | 0.1243 |
| Policy variation across training seeds | 0.00555 | 0.00977 |

Value error is more variable than reward, policy, or Q error. The fixed study
checks use reward recovery, policy recovery, training-seed stability, and
counterfactual regret. The complete value distribution remains in the result
file.

## Sampling Uncertainty

The estimator can also report percentile bootstrap intervals. Resampling is
at the individual-trajectory level. Each draw refits the neural reward map.

The calibration uses 50 generated panels. Each panel has 300 individuals and
one observation per individual. Each panel requests 99 bootstrap draws. The
reward map has three states and two actions. Action 0 is fixed at zero. Reward
and policy coverage is measured at all three states. Counterfactual coverage
uses mean policy movement and mean value change.

All 50 point fits were usable. A total of 4,948 of 4,950 bootstrap refits
converged. Every panel retained at least 98 draws.

| Function | Coverage | Lower-tail miss | Upper-tail miss | Width ratio |
| --- | ---: | ---: | ---: | ---: |
| Reward map | 0.940 | 0.020 | 0.040 | 1.199 |
| Policy | 0.940 | 0.020 | 0.040 | 1.147 |
| Reward change | 0.920 | 0.010 | 0.070 | 1.323 |
| Transition change | 0.920 | 0.050 | 0.030 | 1.300 |
| Action removal | 0.960 | 0.020 | 0.020 | 1.297 |

The width ratio is the largest ratio of the 95th-percentile width to the
median width among the reported quantities in each row. Every interval had
positive width. Two short runs with the same seed produced identical records.
These are generated-data sampling results, not a reproduction of published
paper numbers.

The runnable program is
[`bootstrap_calibration.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/bootstrap_calibration.py).

```bash
PYTHONPATH=src:. python \
  validation/estimators/deep_mce_irl/bootstrap_calibration.py \
  --checkpoint validation/results/deep_mce_irl_bootstrap_calibration.jsonl \
  --output validation/results/deep_mce_irl_bootstrap_calibration.json \
  --quiet --jobs 2
```

**Result**

```text
wrote validation/results/deep_mce_irl_bootstrap_calibration.json
status: ready
usable panels: 50/50
minimum successful draws: 98/99
```

## Counterfactual Design

For each intervention, the fitted counterfactual is compared with an oracle
solution computed under the same change and the simulator's true reward map.

| Family | Changed primitive |
| --- | --- |
| Reward | A payoff component |
| Transitions | The stochastic transition kernel |
| Action availability | One action removed from the choice set |

Each intervention must change the oracle policy. Recovery is measured by
welfare regret under the changed environment.

| Family | Median regret | 95th percentile | Minimum oracle policy movement |
| --- | ---: | ---: | ---: |
| Reward | 0.00474 | 0.01308 | 0.02631 |
| Transitions | 0.00427 | 0.01098 | 0.00835 |
| Action availability | 0.00445 | 0.01305 | 0.28971 |

All six regret checks passed. All three oracle-movement checks also passed, so
low regret does not come from an intervention that leaves the oracle policy
unchanged.

The runnable program is
[`ready.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/ready.py).
It writes one JSONL record per fit and resumes completed records after an
interruption.

```bash
PYTHONPATH=src:. python validation/estimators/deep_mce_irl/ready.py \
  --checkpoint validation/results/deep_mce_irl_ready.jsonl \
  --output validation/results/deep_mce_irl_ready.json \
  --quiet
```

**Result**

```text
wrote validation/results/deep_mce_irl_ready.json
status: ready
```
