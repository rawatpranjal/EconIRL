# Simulation Studies

EconIRL simulation-study pages use synthetic environments. The
harness fixes the reward, transition law, policy, value function, Q function,
and counterfactual oracle before fitting the estimator.

The tables report more than imports or numeric outputs. They state which
objects are checked for each estimator target.

## Numerical Checks

| Numerical check | What it checks |
| --- | --- |
| Parameter recovery | Structural parameter direction and scale where identified. |
| Reward recovery | The fitted reward surface after the estimator normalization. |
| Policy recovery | The total variation distance from the oracle policy. |
| Value and Q recovery | The dynamic objects implied by the recovered reward. |
| Type A counterfactual | A payoff intervention with transitions held fixed. |
| Type B counterfactual | A transition intervention with rewards held fixed. |
| Type C counterfactual | An action-set intervention. |

## How to Read the Results

| Status | Meaning |
| --- | --- |
| Results reported | The page reports the simulation design, results file, and numerical checks. |
| Results with scope | The reported results are tied to stated simulation conditions. |
| Research preview | The implementation is available for experiments; current results are narrower than the target. |

Estimator pages link to results files when machine-readable results are available.
When the results are scoped, the page states the current use clearly.
