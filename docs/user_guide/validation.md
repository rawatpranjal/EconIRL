# Validation

EconIRL validation claims are based on known-truth synthetic environments. The
validation harness knows the reward, transition law, policy, value function,
Q function, and counterfactual oracle before fitting the estimator.

This is stricter than checking that an estimator imports or produces a numeric
answer. A validated structural estimator recovers the objects required by its
target use case.

## Gate Families

| Gate family | What it checks |
| --- | --- |
| Parameter recovery | Structural parameter direction and scale where identified. |
| Reward recovery | The fitted reward surface after the estimator normalization. |
| Policy recovery | The total variation distance from the oracle policy. |
| Value and Q recovery | The dynamic objects implied by the recovered reward. |
| Type A counterfactual | A payoff intervention with transitions held fixed. |
| Type B counterfactual | A transition intervention with rewards held fixed. |
| Type C counterfactual | An action-set intervention. |

## How to Read Status Labels

| Status | Meaning |
| --- | --- |
| Validated | The target passes the documented validation gates. |
| Validated with scope | The target passes under the stated conditions. |
| Research preview | The implementation is available for experiments while evidence is still narrower than the full structural claim. |

Estimator pages link to artifacts when machine-readable evidence is available.
When the evidence is scoped, the page states the recommended use clearly.
