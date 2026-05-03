# GCL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Finn, Levine, and Abbeel (2016) | Neural | Optional | No | Yes |

## What this estimator does

Maximum entropy IRL learns a cost function that explains expert behavior, but computing the partition function requires solving the MDP at each gradient step. For continuous-state robotics tasks with unknown dynamics, exact computation is infeasible. Finn, Levine, and Abbeel (2016) propose Guided Cost Learning, which estimates the partition function gradient via importance sampling from an adaptively learned policy instead of solving the MDP exactly. The policy is updated by reinforcement learning to track the current cost, producing lower-variance importance weights as training progresses.

GCL uses a neural network to parameterize the cost function, removing the need for hand-crafted features. Like all MaxEnt IRL methods, GCL identifies the cost function only up to additive constants and potential-based shaping. Unlike AIRL, GCL does not structurally disentangle the reward from the shaping potential, so the recovered cost is not guaranteed to transfer across environments. The recovered cost has no interpretable parameters and does not support standard errors.

## How it works

The gradient of the IOC objective has the form

$$
\frac{\partial \mathcal{L}_{\text{IOC}}}{\partial \theta} = \frac{1}{N}\sum_{i} \nabla_\theta c_\theta(\tau_i^{\text{demo}}) - \sum_j \tilde{w}_j \nabla_\theta c_\theta(\tau_j^{\text{samp}}),
$$

where $\tilde{w}_j$ are normalized importance weights computed from the ratio of the model distribution to the sampling policy. Each iteration samples trajectories from the current policy, computes importance weights, updates the cost network by gradient descent, and then updates the policy via soft value iteration on the current cost. Importance weight clipping caps the influence of any single trajectory to prevent variance explosion.

## When to use it

GCL is most valuable in continuous-state, continuous-action settings where neither exact value iteration nor feature engineering is feasible. On tabular problems with linear rewards, MCE-IRL is strictly better because it estimates fewer parameters and produces standard errors. If transferable rewards are needed, use AIRL. If the state space is discrete and small, Deep MCE-IRL is more precise because it avoids importance sampling variance. GCL's advantage appears on problems where the true cost is nonlinear, the correct features are unknown, and interpretability is not required.

## References

- Finn, C., Levine, S., and Abbeel, P. (2016). Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization. *ICML 2016*.

The full derivation, algorithm, and simulation results are in the [GCL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/gcl.pdf).
