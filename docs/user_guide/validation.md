# Validation

EconIRL validation pages use known-truth synthetic environments. The harness
knows the reward, transition law, policy, value function, Q function, and
counterfactual oracle before fitting the estimator.

The tables report more than imports or numeric outputs. They state which
objects are checked for each estimator target.

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
| Evidence reported | The page reports gates for the stated target. |
| Evidence with scope | The reported gates are tied to stated conditions. |
| Research preview | The implementation is available for experiments; current evidence is narrower than the target. |

Estimator pages link to artifacts when machine-readable evidence is available.
When the evidence is scoped, the page states the current use clearly.
