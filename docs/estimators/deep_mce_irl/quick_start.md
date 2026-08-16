# Quick Start

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Simulation Study](validation.md)
- [Wulfmeier-Shaped Study](wulfmeier_objectworld.md)
- [Counterfactuals](counterfactuals.md)

Neural MCE-IRL learns an anchored reward map from demonstrations. It needs a
known transition tensor in `(n_actions, n_states, n_states)` orientation.

## Generate Demonstrations

```python
import numpy as np
import pandas as pd

transitions = np.array(
    [
        [[0.85, 0.15, 0.0], [0.0, 0.85, 0.15], [0.15, 0.0, 0.85]],
        [[0.15, 0.85, 0.0], [0.0, 0.15, 0.85], [0.85, 0.0, 0.15]],
    ]
)
expert_policy = np.array([[0.80, 0.20], [0.55, 0.45], [0.25, 0.75]])

rng = np.random.default_rng(7)
records = []
for person in range(700):
    state = person % 3
    for period in range(40):
        action = rng.choice(2, p=expert_policy[state])
        records.append((person, period, state, action))
        state = rng.choice(3, p=transitions[action, state])
data = pd.DataFrame(
    records,
    columns=["person", "period", "state", "action"],
)
print(data.shape)
```

**Result**

```text
(28000, 4)
```

The generated panel follows one fixed expert policy under known stochastic
dynamics. The seed fixes the sampled actions and state sequence.

## Fit the Reward Map

```python
import jax

from econirl.estimators import MCEIRLNeural

model = MCEIRLNeural(
    n_states=3,
    n_actions=2,
    discount=0.9,
    anchor_action=0,
    state_encoder=lambda states: jax.nn.one_hot(states, 3),
    state_dim=3,
    reward_hidden_dim=8,
    reward_num_layers=1,
    max_epochs=300,
    lr=1e-2,
    seed=7,
)
model.fit(
    data,
    state="state",
    action="action",
    id="person",
    transitions=transitions,
)
print(f"converged={model.converged_}")
print(f"termination={model.termination_reason_}")
print(f"occupancy_residual={model.occupancy_moment_residual_:.6f}")
print(f"reward_shape={model.reward_.shape}")
print(f"policy_shape={model.policy_.shape}")
print(f"policy_row_sums={np.round(model.policy_.sum(axis=1), 6).tolist()}")
```

**Result**

```text
converged=True
termination=converged
occupancy_residual=0.004053
reward_shape=(3, 2)
policy_shape=(3, 2)
policy_row_sums=[1.0, 1.0, 1.0]
```

`reward_` is the anchored reward matrix. `policy_` contains the induced choice
probabilities. The maximum occupancy residual for this fit is 0.004053. Check
`converged_`, `termination_reason_`, and
`occupancy_moment_residual_` before using any fitted policy. Raw network
weights are not an identified economic parameter.

For a state-only reward, set `reward_type="state"`. The estimator subtracts the
reward at `anchor_state=0` from every state by default. It reports one reward
per state.

Feature inputs passed through `features=` provide a descriptive linear
projection of the fitted reward map. `projection_diagnostics_` reports its
rank, condition number, residual scale, and R-squared. The projection does not
provide sampling standard errors.

## Sampling Uncertainty

Set `se_method="bootstrap"` and choose `n_bootstrap` to request sampling
uncertainty. Each draw resamples complete individual trajectories and refits
the reward network. `se_seed` controls resampling. `seed` continues to control
network initialization.

`conf_int()` reports percentile intervals for supported reward cells and policy
probabilities. Fixed anchor cells and unavailable actions are omitted.
`bootstrap_` retains the successful reward and policy draws. `se_` and
`pvalues_` remain `None` because network weights and descriptive projection
coordinates are not structural parameters.

The
[`neural_mce_irl_applied_workflow.ipynb`](https://github.com/rawatpranjal/EconIRL/blob/main/examples/neural-mce-irl/neural_mce_irl_applied_workflow.ipynb)
notebook shows fitting, held-out prediction, bootstrap intervals, a reward
counterfactual, and serialization in one workflow.
