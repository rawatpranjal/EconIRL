# Counterfactuals

Read this page as horizon-specific reward transfer. The same recovered reward is
re-solved under interventions, but the behavioral interpretation still depends
on the planning horizon.

The public `RHIP` wrapper exposes the recovered policy, reward matrix, and
value function. It does not yet provide a one-call `counterfactual` method like
the structural likelihood wrappers.

RHIP recovers a linear reward under the additive-linear-reward assumption
($r_\theta(s, a) = \phi(s,a)^\top \theta$). That parametric form supports
re-solve counterfactuals: change the environment primitives, re-solve the
dynamic program under the recovered reward, and compare the resulting policy.

The counterfactual results come from the simulation harness. The harness reruns
the dynamic program under controlled changes and compares the
recovered-reward policy with the oracle policy. See the
[simulation study page](validation.md) for the generator script, table source,
and results file.

## Counterfactual Families

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Reported Results

These rows come from the high-dimensional route-choice study, averaged over
two replications at $H = 3$.

| Counterfactual | Regret |
| --- | ---: |
| Baseline | 0.0201 |
| Type A | 0.0197 |
| Type B | 0.0136 |
| Type C | 0.0205 |

Regret reports how the policy induced by the recovered reward compares with the
oracle counterfactual policy. Numbers for other horizons are in the
[simulation study table](../../simulation_studies/highdim_route_choice.md).

## IRL Caveat

RHIP is an IRL estimator. The reward is identified only up to
behavior-preserving transformations, including additive constants and reward
shaping. Counterfactual predictions are in the same normalization as the
recovered reward. For predictions that require the true cardinal reward level,
a structural estimator that reports standard errors for $\theta$ is the right
choice.

## API Boundary

For package users, the stable public objects are the fitted reward, policy, and
value arrays (`model.reward_matrix_`, `model.policy_`, `model.value_`). For
controlled payoff, transition, or action-set interventions, use the simulation
and evaluation utilities with an explicit problem and transition environment.
