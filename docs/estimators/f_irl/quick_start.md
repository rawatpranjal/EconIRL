# Quick Start

```python
from econirl.estimation import FIRLEstimator

estimator = FIRLEstimator(
    f_divergence="fkl",      # forward KL (primary validated divergence)
    marginal_space="state",  # match state marginals
    reward_scope="state",    # learn a state-only reward
    lr=0.5,
    max_iter=250,
)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)                          # choice probabilities (n_states, n_actions)
print(summary.value_function)                  # value function V(s)
print(summary.metadata["occupancy_l1"])        # state marginal L1 distance
print(summary.metadata["reward_range"])        # reward range (check for flat-reward failure)
print(summary.metadata["reward_matrix"])       # tabular reward (n_states, n_actions)
```

Fitted attributes returned in `summary.metadata`:

| Key | Meaning |
| --- | --- |
| `reward_matrix` | Tabular reward matrix, shape `(n_states, n_actions)`. State scope tiles one column. |
| `state_reward_vector` | State-only reward vector, shape `(n_states,)`. `None` for state-action scope. |
| `expert_marginal` | Empirical state marginal from the panel. |
| `policy_marginal` | Model state marginal at the selected iterate. |
| `occupancy_l1` | L1 distance between expert and model state marginals. |
| `reward_range` | Range of the recovered reward matrix; near-zero indicates a flat-reward failure. |
| `f_divergence` | Divergence family used. |
| `marginal_space` | Whether state or state-action marginals were matched. |
| `reward_scope` | Whether the learned reward is state-only or state-action. |

## Divergence Choices

```python
estimator = FIRLEstimator(f_divergence="fkl")   # forward KL, mass-covering
estimator = FIRLEstimator(f_divergence="rkl")   # reverse KL, mode-seeking
estimator = FIRLEstimator(f_divergence="js")    # Jensen-Shannon, symmetric
estimator = FIRLEstimator(f_divergence="chi2")  # chi-squared
estimator = FIRLEstimator(f_divergence="tv")    # total variation
```

Forward KL is the primary validated choice. The others are available for
comparison but are not separately validated on the paper-faithful cell.

## Checking for Flat-Reward Failure

A near-zero reward range means the optimizer found a flat reward, which
produces a near-uniform policy and fails as a recovery attempt. Check this
before reading other metrics:

```python
if summary.metadata["reward_range"] < 0.001:
    print("Flat-reward failure: check learning rate, divergence, or marginal scope.")
```

## Lower-Level Control

`FIRLEstimator` takes an explicit `Panel`, utility specification, `DDCProblem`,
and transition tensor, matching the other low-level estimators:

```python
from econirl.estimation import FIRLEstimator

result = FIRLEstimator(
    f_divergence="fkl",
    marginal_space="state",
    reward_scope="state",
    lr=0.5,
    max_iter=250,
    reward_clip=10.0,
).estimate(panel, utility, problem, transitions)
```
