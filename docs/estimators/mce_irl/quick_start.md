# Quick Start

## Important Links

- [MCE-IRL overview](../mce_irl.md)
- [Pre-estimation checks](pre_estimation.md)
- [Simulation evidence](validation.md)
- [Counterfactuals](counterfactuals.md)

MCE-IRL estimates one reward from demonstrated trajectories and a fixed
transition system. A task supplies the start distribution, destination,
horizon, and active part of that system. Different tasks can share the same
reward parameters.

The example below has two task labels with the same destination, dynamics, and
reward. The demonstrations choose action one in 70 percent of routes.

```python
import jax.numpy as jnp
import numpy as np

from econirl import DeterministicTransitions, MCEIRL, MCEIRLTask
from econirl.core import Panel, Trajectory

transitions = DeterministicTransitions(
    next_state=np.array([[1, 1], [1, -1]]),
    valid_action=np.array([[True, True], [True, False]]),
)
features = np.zeros((2, 2, 1))
features[0, 1, 0] = 1.0
tasks = [
    MCEIRLTask(
        task_id=name,
        initial_state=0,
        terminal_states=np.array([1]),
        horizon=1,
    )
    for name in ("morning", "evening")
]
trajectories = [
    Trajectory(
        states=jnp.array([0]),
        actions=jnp.array([int(index < 70)]),
        next_states=jnp.array([1]),
        individual_id=f"{name}-{index}",
        metadata={"task_id": name},
    )
    for name in ("morning", "evening")
    for index in range(100)
]

model = MCEIRL(
    n_states=2,
    n_actions=2,
    discount=1.0,
    horizon=1,
    feature_matrix=features,
    feature_names=["action_one"],
    compute_se=False,
)
model.fit(
    Panel(trajectories),
    transitions=transitions,
    tasks=tasks,
)

print({name: round(value, 6) for name, value in model.params_.items()})
print(
    np.round(
        model.predict_proba(
            np.array([0]),
            task_id="morning",
        ),
        3,
    ).tolist()
)
counterfactual = model.counterfactual(
    params={"action_one": model.params_["action_one"] + 1.0}
)
print(round(float(np.abs(counterfactual.policy_change).max()), 3))
print(model.termination_reason_)
```

**Result**

```text
{'action_one': 0.847298}
[[0.3, 0.7]]
0.164
joint_convergence
```

`next_state[s, a]` stores the deterministic successor of state `s` under
action `a`. `valid_action[s, a]` marks legal actions. Use `-1` for an invalid
successor only when the corresponding action is false. The estimator never
constructs an `(A, S, S)` tensor for this representation.

Each trajectory carries its task identifier in `metadata["task_id"]`. For
data-frame input, pass `state`, `action`, `id`, `next_state`, and `task` to
`fit`. For example, use `fit(data, state="state", action="action",
id="route_id", next_state="next_state", task="task_id")`. State and next-state
values must use the same global indexing as the fixed transition system.

`MCEIRLTask.active_states` can restrict a destination to a compact candidate
path set. The compiler removes transitions that leave that set. It rejects
demonstrations that use an invalid action or disagree with the supplied
successor.

The main fitted attributes are:

| Attribute | Meaning |
| --- | --- |
| `params_` | Shared reward parameters. |
| `se_` | Standard errors when requested. |
| `time_policy_` | Period-specific policy for a finite horizon. |
| `task_policy_` | Policy slices by task. |
| `reward_matrix_` | Fitted state-action reward. |
| `termination_reason_` | Reason the fit stopped. |
| `feature_residual_` | Norm of the final stationarity residual. |

Use `simulate(..., task_id=...)` to draw routes from the fitted model.
Use `counterfactual(params=...)` for a reward change or
`counterfactual(transitions=...)` for a transition change. Exactly one
primitive can change in each call.
