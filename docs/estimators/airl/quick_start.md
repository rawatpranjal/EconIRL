# Quick Start

The low-level `AIRLEstimator` accepts a `Panel`, a utility specification, a
`DDCProblem`, and a transition tensor. The high-level `econirl.estimators.AIRL`
alias points to the neural wrapper; use `AIRLEstimator` for tabular work.

```python
from econirl.estimation import AIRLConfig
from econirl.estimation.adversarial import AIRLEstimator

config = AIRLConfig(
    reward_type="linear",
    reward_arg="state",
    use_shaping=True,
    max_rounds=150,
    reward_lr=0.01,
    discriminator_steps=10,
)
estimator = AIRLEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)       # shape (n_states, n_actions)
print(summary.converged)    # True if policy converged before max_rounds
```

Key fitted objects:

| Attribute | Meaning |
| --- | --- |
| `summary.policy` | Recovered choice probabilities, shape (n_states, n_actions). |
| `summary.parameters` | Learned reward parameters (linear) or reward matrix entries (tabular). |
| `summary.converged` | Whether policy change fell below `convergence_tol` before `max_rounds`. |
| `summary.num_iterations` | Training rounds used. |
| `summary.metadata["final_disc_loss"]` | Final discriminator loss; near log(2) at the adversarial optimum. |

## Reward Argument

```python
# State-only: matches Fu et al. (2018) identification guarantee
config = AIRLConfig(reward_arg="state", ...)

# State-action: needed when the DGP has action-dependent payoffs
# Longer schedule required; increase max_rounds and reward_lr
config = AIRLConfig(
    reward_arg="state_action",
    max_rounds=300,
    reward_lr=0.01,
    discriminator_steps=10,
)
```

The state-only mode projects the learned reward matrix onto the state-only
subspace by averaging across actions before computing the discriminator logit.
The state-action mode lets the reward differ by action, which is needed for
discrete choice problems with action-dependent payoffs but loses the
disentanglement guarantee.

## Reproduce the Simulation

```bash
PYTHONPATH=src:. python validation/estimators/airl/run.py --quiet-progress
```
