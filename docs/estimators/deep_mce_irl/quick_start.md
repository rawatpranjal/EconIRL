# Quick Start

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)

Neural MCE-IRL learns an anchored reward map from demonstrations. It needs a
known transition tensor in `(n_actions, n_states, n_states)` orientation.

```python
import numpy as np
import pandas as pd

from econirl.estimators import MCEIRLNeural

transitions = np.array(
    [
        [[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
    ]
)
data = pd.DataFrame(
    {
        "person": [0, 0, 0, 1, 1, 1],
        "state": [0, 1, 2, 0, 1, 2],
        "action": [0, 0, 0, 1, 0, 0],
    }
)

model = MCEIRLNeural(
    n_states=3,
    n_actions=2,
    anchor_action=0,
    max_epochs=20,
    seed=7,
)
model.fit(
    data,
    state="state",
    action="action",
    id="person",
    transitions=transitions,
)
print(model.reward_.shape)
print(model.policy_.shape)
print(np.round(model.policy_.sum(axis=1), 6).tolist())
```

**Result**

```text
(3, 2)
(3, 2)
[1.0, 1.0, 1.0]
```

`reward_` is the anchored reward matrix. `policy_` contains the induced choice
probabilities. This six-observation example demonstrates the API and array
shapes. It does not meet the estimator's convergence test. Check `converged_`,
`termination_reason_`, and `occupancy_moment_residual_` before using a fitted
policy. Raw network weights are not an identified economic parameter.

For a state-only reward, set `reward_type="state"`. The estimator subtracts the
reward at `anchor_state=0` from every state by default. It reports one reward
per state.

Feature inputs passed through `features=` provide a descriptive linear
projection of the fitted reward map. `projection_diagnostics_` reports its
rank, condition number, residual scale, and R-squared. The projection does not
provide sampling standard errors.
