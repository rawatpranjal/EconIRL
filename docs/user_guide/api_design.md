# API Design

EconIRL exposes two layers. The public layer accepts ordinary panel
DataFrames and returns fitted attributes in a sklearn-style object. The
research layer accepts structured model objects when the caller needs direct
control over transitions, utilities, state encodings, or diagnostic metadata.

## Public DataFrame Path

Use the public estimator classes when the data already fit a finite-state
dynamic choice problem. The caller supplies the DataFrame and names the state,
action, and individual identifier columns.

```python
from econirl import NFXP

model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

model.params_
model.policy_
model.value_
model.summary()
```

The public path keeps the common workflow short: construct the estimator, call
`fit`, then inspect parameters, policies, values, and summaries.

## Data Contract

The DataFrame path assumes one row per observed decision. The state and action
columns must use integer labels or values that can be mapped cleanly to finite
integer labels. The identifier column groups decisions by agent or trajectory.

| Input | Role |
| --- | --- |
| `state` | Current decision state. |
| `action` | Observed discrete action. |
| `id` | Agent, trajectory, or panel identifier. |
| `next_state` | Optional observed next state when supplied by the workflow. |
| `transitions` | Optional transition tensor when the caller does not want first-stage estimation. |
| `features` | Optional reward-feature array for estimators that need supplied features. |

When transitions are not supplied, tabular estimators estimate them from the
observed state, action, and next-state records. When reward features are
required, the feature shape must match the estimator's state-action indexing.

## Lower-Level Research Path

The lower-level `econirl.estimation` namespace is for workflows that already
construct model objects directly. It is useful when a paper replication or
simulation study needs exact control over the panel container, transition
tensor, utility object, optimization settings, or returned metadata.

```python
from econirl.estimation import NFXPEstimator

estimator = NFXPEstimator(config=config)
summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)
```

This layer is less compact, but it makes the model boundary explicit: the
caller provides the economic problem, the transition law, and the reward or
utility representation.

## Fitted Outputs

Public estimators use common fitted attribute names where possible.

| Output | Meaning |
| --- | --- |
| `params_` | Estimated reward or utility parameters. |
| `standard_errors_` | Standard errors when the estimator reports them. |
| `policy_` | Estimated conditional choice probabilities or policy table. |
| `value_` | Estimated value function when the estimator computes one. |
| `q_values_` | Choice-specific values when available. |
| `log_likelihood_` | Maximized likelihood or pseudo-likelihood for likelihood estimators. |
| `summary()` | Human-readable report of estimates and diagnostics. |

Estimator-specific pages document outputs that do not fit this shared surface,
such as learned neural reward tables, projected reward parameters, or segment
membership summaries.

## Namespaces

The top-level `econirl` and `econirl.estimators` imports are the user-facing
entry points. They prioritize stable examples and compact workflows.

The `econirl.estimation` namespace exposes the lower-level estimators used by
simulation studies and research examples. Use it when the workflow needs
structured objects instead of DataFrame column names.

## Simulation Study Interface

Simulation-study scripts use the lower-level API because they need precise
control over the synthetic MDP, solver settings, and reported diagnostics. The
public estimator pages then translate those results back into the user-facing
API surface.

This separation keeps quick examples readable while preserving an exact path
from the reported simulation to the estimator implementation.

## Design Rule

The public API should make the main workflow easy to find. The lower-level API
should expose enough structure for research workflows without changing the
simple DataFrame path.
