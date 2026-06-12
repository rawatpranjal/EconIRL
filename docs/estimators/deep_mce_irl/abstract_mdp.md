# Abstract MDP Example

The bundled abstract MDP generator gives a quick end-to-end smoke test:
build a synthetic environment with a known neural reward, simulate a panel,
fit Deep MCE-IRL, and inspect the recovered reward matrix and policy.

```python
import numpy as np
from econirl.environments import ArrayMDP
from econirl.estimators import MCEIRLNeural

# Build a small synthetic environment (supply your own transitions and reward)
n_states, n_actions = 16, 3
discount = 0.95

# transitions: (n_actions, n_states, n_states), row-stochastic
transitions = np.load("transitions.npy")

# Simulate a panel from the true policy
# (replace with your own DataFrame of state/action/id columns)
df = ...

# Fit Deep MCE-IRL with an anchor on action 0
model = MCEIRLNeural(
    n_states=n_states,
    n_actions=n_actions,
    discount=discount,
    reward_type="state_action",
    anchor_action=0,
    reward_hidden_dim=32,
    reward_num_layers=2,
    max_epochs=300,
    lr=3e-3,
)
model.fit(
    data=df,
    state="state",
    action="action",
    id="agent_id",
    transitions=transitions,
)

print("Converged:", model.converged_)
print("Reward matrix shape:", model.reward_.shape)   # (16, 3)
print("Policy TV vs uniform:", ...)
print(model.summary())
```

## Interpretation

The `reward_` attribute holds the anchored neural reward matrix $R(s, a)$.
Action 0 has zero reward everywhere by construction (the anchor). The
remaining columns represent the relative value of actions 1 and 2 versus
action 0 at each state. Because the reward is anchored, the matrix can be
compared against an oracle reward computed under the same gauge.

The `policy_` and `value_` attributes come from solving the soft Bellman
equation at the final learned reward, so they are the policy and value the
recovered reward actually implies - not a fit to the data directly.

## Replication Boundary

This page is a package smoke test on a synthetic environment, not a full
historical replication of any published study. The estimator's recovery
properties are established on the primary synthetic cell whose
data-generating process is fully specified; see the
[Simulation Study](validation.md) page. The
[bus engine](../../simulation_studies/rust_bus.md) and
[simulation studies](../../simulation_studies/index.md) pages compare Deep
MCE-IRL against the full estimator roster on shared benchmarks.
