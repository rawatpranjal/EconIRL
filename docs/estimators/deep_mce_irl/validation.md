# Simulation Study

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Wulfmeier-Shaped Study](wulfmeier_objectworld.md)
- [Counterfactuals](counterfactuals.md)
- [Result file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl_ready.json)

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

All 300 fits completed and met the estimator convergence test. All 18 study
checks passed.

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

## Counterfactual Design

Three interventions are re-solved against their oracle versions.

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
PYTHONPATH=src:. python validation/estimators/deep_mce_irl/ready.py --quiet
```

**Result**

```text
wrote validation/results/deep_mce_irl_ready.json
status: ready
```
