# Problem Setup and API Design

EconIRL uses a sklearn-style surface for public estimator workflows. You pass a
panel DataFrame, name the state, action, and individual identifier columns, and
then read fitted attributes from the estimator.

```python
from econirl import NFXP

model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

model.params_
model.policy_
model.value_
```

The lower-level `econirl.estimation` namespace remains available for research
workflows that need direct access to panel objects, transition tensors, utility
objects, and diagnostic metadata.

## Public Objects

| Object | Role |
| --- | --- |
| DataFrame input | The common path for examples and quickstarts. |
| `Panel` input | The structured path for advanced workflows. |
| `TransitionEstimator` | The first-stage transition estimator for tabular DDC. |
| `NFXP`, `CCP`, `NNES`, `TDCCP` | Public sklearn-style estimator classes. |
| `summary()` | A compact report with estimates, standard errors, and fit output. |

## Design Rule

The public API keeps the main evidence-backed path easy to find. Research
estimators remain available for advanced users, and their pages describe the
scope of their current evidence.
