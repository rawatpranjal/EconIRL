# Rust Bus Engine Example

The Rust bus-engine replacement problem is a useful smoke example for MCE-IRL,
but it is not the release validation cell. The wrapper needs an explicit reward
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

Output

```text
No fixed output. This is a smoke fit, and estimates depend on the chosen
action-dependent feature matrix and transition specification.
```

## Interpretation

The first feature assigns a mileage cost to keeping the engine. The second
feature assigns a replacement cost to the replacement action. The fitted policy
gives replacement probabilities by mileage state.

## Validation Boundary

This page is not a historical Rust replication and is not the certified
evidence claim. The release evidence comes from the known-truth MCE cell, where
reward, transition, policy, value, Q, and counterfactual oracle objects are all
known.
