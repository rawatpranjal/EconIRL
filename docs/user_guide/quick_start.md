# Quick Start

Install the package, fit the reference estimator on the Rust bus data, and read
the structural parameters.

## Install

```bash
pip install econirl
```

## Fit NFXP

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()
model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
```

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
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

```python
cf = model.counterfactual(RC=4.0)
print(cf.policy[50, 1])
```

```text
0.05519477716656161
```

Next, check the data and the design before trusting a fit. See
[Loading Data](your_own_data.md) and
[Pre-Estimation Diagnostics](pre_estimation.md).
