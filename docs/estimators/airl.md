# AIRL

## Important Links

- [Quick Start](airl/quick_start.md)
- [Pre-Estimation Checks](airl/pre_estimation.md)
- [Identification Boundary](airl/identification.md)
- [Taxi Dynamics Transfer](airl/taxi_transfer.md)
- [Applied Notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/airl/airl_applied_workflow.ipynb)

Adversarial inverse reinforcement learning estimates a reward from observed
state, action, and next-state triples. The public `AIRL` class implements the
tabular method of Fu, Luo, and Levine (2018). It constrains the reward to depend
on state alone.

This restriction matters. It is the setting in which the discriminator can
separate a reward component from dynamics-dependent shaping. The recovered
reward can then be re-solved after the transition system changes.

## Source Papers

The estimator follows {ref}`Fu, Luo, and Levine (2018) <fu-2018>`. The paper
introduces the structured discriminator used to separate a state reward from a
dynamics-dependent potential term.

## Model

Let $s$ denote state, $a$ action, and $s'$ next state. AIRL uses the structured
discriminator score

$$
f_{\theta,\phi}(s,a,s') = g_\theta(s) + \beta h_\phi(s') - h_\phi(s).
$$

The discriminator is

$$
D(s,a,s') =
\frac{\exp f_{\theta,\phi}(s,a,s')}
{\exp f_{\theta,\phi}(s,a,s') + \pi(a\mid s)}.
$$

The state reward $g_\theta$ is the transferable component. The potential
$h_\phi$ absorbs continuation terms that depend on the original dynamics.
During fitting, the policy update uses the configured discriminator-derived
reward. The reported policy and value are then re-solved from the learned state
reward.

## Algorithm

```text
initialize reward, potential, and policy
repeat
    compare demonstration transitions with samples from the current policy
    update the structured discriminator
    update the policy from the configured discriminator reward
until the policy-change rule is met or maximum rounds are reached
re-solve policy and value from the learned state reward
```

## Identification

The reward interpretation requires all of the following.

- The systematic reward is state-only.
- The MDP satisfies the decomposability condition in Fu et al. (2018).
- The transition tensor is correctly specified.
- The declared state feature matrix has full column rank.
- The adversarial fit reaches its stopping rule.

Under these conditions, the reward is recovered up to an additive constant.
`reward_` is centered over states. Raw adversarial weights are not structural
coefficients.

The public estimator rejects action-dependent reward features. It also rejects
`context=`. Those inputs require a different identification design. See the
[Identification Boundary](airl/identification.md).

## Public workflow

The recommended import is `from econirl import AIRL`. Use
[`NeuralAIRL`](neural_airl.md) when the state reward needs nonlinear function
approximation.

`fit` accepts a `Panel`, `TrajectoryPanel`, or a DataFrame. DataFrame input
requires state, action, individual, and next-state columns. The transition
tensor must have orientation `(n_actions, n_states, n_states)`.

The fitted model exposes the following objects.

| Object | Meaning |
| --- | --- |
| `reward_` | Centered state reward, shape `(n_states,)`. |
| `reward_matrix_` | The same state reward repeated over actions. |
| `policy_` | Recovered choice probabilities. |
| `value_` | Soft value function computed from the learned state reward before centering. |
| `diagnostics_` | Data, identification, transition, and optimization checks. |
| `bootstrap_` | Trajectory-bootstrap reward and policy functionals, when requested. |

Prediction, simulation, changed-dynamics counterfactuals, summaries, and pickle
serialization use the same fitted object.

The [Simulation Study](airl/validation.md) reports recovery and interval
results. [Counterfactuals](airl/counterfactuals.md) defines the supported
changed-dynamics and reward-parameter scenarios.

## Evidence

The controlled study uses 16 states, 4 actions, 4 state features, and 24,000
observations per replication. All three fits converged.

| Metric | Result |
| --- | ---: |
| Reward normalized RMSE, median | 0.1397 |
| Policy total variation, 95th percentile | 0.0083 |
| Value normalized RMSE, 95th percentile | 0.1523 |
| Q normalized RMSE, 95th percentile | 0.1646 |
| Changed-dynamics policy TV, 95th percentile | 0.0101 |
| Changed-dynamics regret, 95th percentile | 0.0070 |

The trajectory-bootstrap study used 20 independent panels and 19 resamples per
panel. All 380 resampled fits succeeded. Across these generated panels,
empirical interval coverage was 0.900 for centered reward cells and 0.900 for
policy probabilities.

The generated taxi study changes directional reliability and closes an
eastbound corridor. The oracle policy changes by 0.1095 total variation. AIRL
re-solves the recovered reward with 0.0525 transfer policy TV at the 95th
percentile. See [Taxi Dynamics Transfer](airl/taxi_transfer.md).

These are generated adversarial studies. They are not an exact replication of
published paper numbers. Fu et al. Section 7.1 starts from MaxEnt IRL, not the
sampled adversarial AIRL implementation used here.

## Study files

- [Controlled recovery results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl_controlled_recovery.json)
- [Bootstrap results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl_bootstrap_calibration.json)
- [Taxi transfer results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl_taxi_transfer.json)

```{toctree}
:hidden:

airl/quick_start
airl/pre_estimation
airl/validation
airl/counterfactuals
airl/identification
airl/taxi_transfer
```
