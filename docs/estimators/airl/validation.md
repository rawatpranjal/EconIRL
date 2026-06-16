# Simulation Study

AIRL runs on two synthetic cells. The primary cell is `airl_paper_identification`,
a state-only reward DGP designed to match the original identification conditions
of Fu et al. (2018). The second cell is `airl_anchor_action_dependent`, an
action-dependent DGP with an absorbing state and an anchored exit action; it is
a diagnostic cell run to confirm where the method fails, not a validated use
case.

In both cells, the transitions, reward, policy, value function, Q function, and
Type A, Type B, and Type C counterfactual oracles are chosen before generating
the panel. The estimator sees only the demonstrations, the transitions, and the
reward feature basis. The true objects are held back for evaluation.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/airl/run.py).
It writes the results file
[`airl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl.json).

```bash
PYTHONPATH=src:. python validation/estimators/airl/run.py --quiet-progress
```

## Design

| Quantity | Primary cell | Diagnostic cell |
| --- | ---: | ---: |
| Cell id | `airl_paper_identification` | `airl_anchor_action_dependent` |
| States | 16 | 21 |
| Actions | 4 | 3 |
| Individuals | 300 | 300 |
| Periods per individual | 80 | 80 |
| Observations | 24,000 | 24,000 |
| Reward arg | state | state_action |
| Shaping | learned | learned |

## Fit Summary

Primary cell only; the diagnostic cell is discussed under [Numerical Checks](#numerical-checks).

| Quantity | Value |
| --- | --- |
| Converged | true |
| Log-likelihood | -33173.0703 |
| Training rounds | 150 |
| Estimation time | 56.53 seconds |
| Final discriminator loss | 1.386 |

## Recovery Metrics

| Metric | Primary cell |
| --- | ---: |
| Reward NRMSE | 0.0998 |
| Policy TV | 0.0060 |
| Value NRMSE | 0.1099 |
| Q NRMSE | 0.1201 |

Reward NRMSE is the normalized root mean squared error of the recovered reward
map projected onto the true reward's range. Policy TV is total-variation
distance from the oracle policy. These are the right scorecards for adversarial
IRL; raw parameter RMSE is not, because reward is identified only up to
potential-based shaping.

## Counterfactual Metrics

| Counterfactual | Policy TV | Regret |
| --- | ---: | ---: |
| Type A | 0.0062 | 0.0029 |
| Type B | 0.0076 | 0.0038 |
| Type C | 0.0075 | 0.0050 |

Type A shifts payoffs and holds transitions fixed. Type B changes transitions
and holds the recovered reward fixed. Type C removes one action. Regret is the
welfare loss from following the recovered-reward policy instead of the oracle
policy in the counterfactual world.

## Numerical Checks

Primary cell (all pass):

| Check | Value | Threshold | Status |
| --- | ---: | --- | --- |
| converged | true | is true | pass |
| reward_normalized_rmse | 0.0998 | at most 0.15 | pass |
| policy_tv | 0.0060 | at most 0.05 | pass |
| value_normalized_rmse | 0.1099 | at most 0.15 | pass |
| q_normalized_rmse | 0.1201 | at most 0.15 | pass |
| type_a_regret | 0.0029 | at most 0.08 | pass |
| type_b_regret | 0.0038 | at most 0.08 | pass |
| type_c_regret | 0.0050 | at most 0.08 | pass |

Diagnostic cell `airl_anchor_action_dependent` (all fail):

| Check | Value | Threshold | Status |
| --- | ---: | --- | --- |
| converged | false | is true | fail |
| reward_normalized_rmse | 1.1606 | at most 0.15 | fail |
| policy_tv | 0.4030 | at most 0.05 | fail |
| value_normalized_rmse | 1.9954 | at most 0.15 | fail |
| q_normalized_rmse | 1.2601 | at most 0.15 | fail |
| type_a_regret | 10.5004 | at most 0.08 | fail |
| type_b_regret | 13.9730 | at most 0.08 | fail |
| type_c_regret | 4.4053 | at most 0.08 | fail |

The diagnostic failure is structural to the method, not a tuning problem. With
a state-only reward, the discriminator cannot represent the action contrast that
drives behavior in an action-dependent DGP. The identification page explains the
boundary in detail.

AIRL also appears in the [bus engine](../../simulation_studies/rust_bus.md) and
[gridworld](../../simulation_studies/taxi_gridworld.md) simulation studies,
where it is compared against the full estimator roster.
