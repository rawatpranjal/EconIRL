# Rust Bus Engine Example

Read this page as a feature-specified smoke test. MCE-IRL depends on the reward
features supplied by the user, so this is an example of wiring, not a canonical
paper replication.

The Rust bus-engine replacement problem is a useful smoke example for MCE-IRL,
but it is not the simulation study. The wrapper needs an explicit reward
feature matrix for multi-action recovery.

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
print(model.predict_proba([0, 10, 50]))
```

Estimates depend on the chosen action-dependent feature matrix and transition
specification, so no canonical output is shown here.

## Interpretation

The first feature assigns a mileage cost to keeping the engine. The second
feature assigns a replacement cost to the replacement action. The fitted policy
gives replacement probabilities by mileage state.

## Replication Boundary

This page is a package smoke test on the bundled dataset, not a full
historical replication of the original study. The estimator's recovery
properties are established on a synthetic cell whose data-generating process
is fully specified; see the [Simulation Study](validation.md) page. The
[bus engine simulation page](../../simulation_studies/rust_bus.md) compares
MCE-IRL against the full estimator roster on a synthetic bus engine panel.
