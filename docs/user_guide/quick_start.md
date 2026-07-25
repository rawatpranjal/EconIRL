# Quick Start

Install the package, fit the reference estimator on the Rust bus data, and read
the structural parameters.

This is a mechanical first run. It shows the package shape on a known example;
do the data and estimator checks before treating a result as evidence for a new
problem.

## Install

```bash
pip install econirl
```

## Fit NFXP

NFXP is used here because the bundled bus example is the canonical small
tabular dynamic discrete-choice problem.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NFXP

df = load_rust_bus()
model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
```

```text
{'operating_cost': 0.001002924937407198, 'replacement_cost': 3.072263682263484}
```

## Read the results

The fitted model exposes the reward parameters, their standard errors, the value
function, and the implied policy.

```python
print(model.summary())
print(model.se_)
```

## Run a counterfactual

Change a structural primitive and read the new policy.
This only makes sense for estimators that recover a reward object the model can
re-solve under the changed primitive.

```python
cf = model.counterfactual(replacement_cost=4.0)
print(cf.policy[50, 1])
```

```text
0.055196266692073837
```

Next, check the data and the design before trusting a fit. See
[Loading Data and Pre-Estimation](your_own_data.md).
