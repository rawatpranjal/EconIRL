# Route Choice Example

The route-choice problem is the canonical setting for RHIP. A traveller moves
through a road network one step at a time, choosing among the nearest neighbours
of the current node. The utility of an edge depends on its length, the amenity
of the destination, and how close the destination sits to a fixed goal node.
This is a smoke example for the wrapper, not the simulation study.

```python
import numpy as np

from econirl import RHIP
from econirl.environments import road_network

# Build a small 25-node road network.
env = road_network(num_nodes=25, num_actions=4, discount_factor=0.95, seed=0)
panel = env.simulate(n_agents=200, n_periods=30, seed=42)

# Features: (n_states, n_actions, 3) -- edge_cost, amenity, goal distance.
features = env.feature_matrix              # shape (25, 4, 3)
transitions = env.transition_matrices      # shape (4, 25, 25)

model = RHIP(
    horizon=3,
    discount=0.95,
    scale=1.0,
    feature_names=["edge_cost", "amenity", "goal"],
)
model.fit(panel, features=features, transitions=transitions)

print(model.params_)
print(model.predict_proba([0, 5, 12]))
```

Estimates depend on the chosen feature matrix, transition specification, and
horizon, so no canonical output is shown here.

## Interpretation

The three features capture the main trade-offs in route choice. `edge_cost` is
the negative Euclidean length of the edge, so a higher weight makes shorter
edges more attractive. `amenity` is a node-level draw, capturing desirability
of the destination. `goal` is the negative shortest-path distance to the
destination node, so a higher weight pulls the traveller toward the goal.

The fitted policy gives action probabilities at each node. Nodes near the goal
have high probability mass on the goal-approaching action. The `horizon` knob
controls how far ahead the agent plans: at `horizon=0` the policy is the
softmax over a deterministic continuation value, at `horizon=float("inf")` it
is the full Max Causal Entropy solution.

## Replication Boundary

This page is a package smoke test on a small synthetic network, not a full
replication of the original study. The estimator's recovery properties are
established on larger synthetic panels where the data-generating process is
fully specified. See the [Simulation Study](validation.md) page. The
[high-dimensional route-choice study](../../simulation_studies/highdim_route_choice.md)
compares RHIP against the full estimator roster on a 150-node network.
