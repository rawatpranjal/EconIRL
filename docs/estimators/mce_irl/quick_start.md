# Quick Start

The wrapper follows the sklearn-style pattern: build an estimator, call `fit`,
then read fitted attributes. For multi-action MCE-IRL, provide reward features
explicitly.

```python
import numpy as np

from econirl.datasets import load_rust_bus
from econirl.estimators import MCEIRL

n_states = 90
n_actions = 2
features = np.zeros((n_states, n_actions, 2))
features[:, 0, 0] = -np.arange(n_states) / 100.0
features[:, 1, 1] = -1.0

df = load_rust_bus()

model = MCEIRL(
    n_states=n_states,
    n_actions=n_actions,
    discount=0.99,
    feature_matrix=features,
    feature_names=["keep_mileage_cost", "replace_cost"],
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.policy_.shape)
```

For a problem other than the Rust bus, pass the dynamics explicitly. Supply a
transition tensor of shape `(n_actions, n_states, n_states)` and the observed
next-state column.

```python
model.fit(
    df,
    state="state",
    action="action",
    id="id",
    next_state="next_state",
    transitions=transitions,
)
```

`transitions=None` estimates only the two-action Rust-bus keep/replace kernel. A
model with more than two actions requires an explicit tensor. Build one from
observed transitions with `estimate_empirical_transitions(panel, n_actions,
n_states)` from `econirl.estimators`.

Estimates depend on the supplied reward features, transition specification, and
inference settings, so no canonical output is shown here. The fitted estimator
exposes reward parameters, standard errors when requested,
the recovered reward, the policy, the value function, and feature-matching
diagnostics.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated reward parameters. |
| `se_` | Standard errors for the reward parameters when available. |
| `reward_matrix_` | Structural reward matrix by state and action. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `log_likelihood_` | Log likelihood of the demonstrations under the recovered policy. |

## Simulation Rerun

To reproduce the simulation, run the lower-level simulation harness:

```bash
PYTHONPATH=src:. python validation/estimators/mce_irl/run.py --quiet-progress --enforce-gates
```

Output

```text
hard gates (canonical_low_action): 10 pass, 0 fail
hard gates (mce_low_high_reward): 10 pass, 0 fail
hard gates total: 20 pass, 0 fail
```

Use `econirl.estimation.mce_irl.MCEIRLEstimator` when you need direct control
over `Panel` objects, utility objects, `DDCProblem`, transition tensors, the
root feature-matching optimizer, or standard-error computation.
