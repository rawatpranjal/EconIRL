# Using your own data

EconIRL fits a panel of decisions. Each row is one observed choice: who decided,
the state they were in, and the action they took. This page takes a panel from
raw data to a fitted model and one counterfactual.

The examples use the bundled bus-engine dataset as a stand-in for your own panel.
Replace it with your DataFrame and the rest stays the same.

## What your data needs

One row per observed decision. Three columns name the pieces.

| Column | What it holds |
| --- | --- |
| `state` | The state when the decision was made, as an integer label. |
| `action` | The discrete action that was chosen, as an integer label. |
| `id` | The individual, unit, or trajectory the row belongs to. |

A `next_state` column is optional. When it is missing, the tabular estimators
read transitions from the order of states within each `id`.

## Two ways to call econirl

Most users hand econirl a DataFrame and name the columns.

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()                        # your DataFrame goes here
model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
```

If you are replicating a paper or running a simulation study, you can build the
model from transition and feature arrays instead. The estimator pages under
[Estimators](../estimators.md) show that path where it applies.

## Check the data before you fit

First, confirm the panel is well formed.

```python
from econirl.preprocessing import check_panel_structure

report = check_panel_structure(df, id_col="bus_id", period_col="period",
                               state_col="mileage_bin", action_col="replaced")
print(report.valid, report.n_individuals, report.n_observations)
print(report.warnings)
```

```text
True 90 9410
['Unbalanced panel: 80-120 periods per individual']
```

The panel is usable. The warning notes that individuals are observed for
different lengths, which these estimators allow.

Next, look for states where only one action ever appears.

```python
counts = df.groupby("mileage_bin")["replaced"].nunique()
print((counts < 2).sum(), "of", df["mileage_bin"].nunique(), "states show one action")
```

```text
14 of 52 states show one action
```

NFXP still fits these states from the likelihood. The choice-probability
estimators, CCP and UFXP, read action frequencies directly, so they need states
with some variation.

Last, check that your reward features can be recovered from choices. A feature
that never changes with the action cancels out of every choice comparison, so
its parameter is not identified. `feature_diagnostics` reports this.

```python
import numpy as np
from econirl.preprocessing import feature_diagnostics

S, A = 20, 3
rng = np.random.default_rng(0)
phi = np.zeros((S, A, 2))
phi[:, :, 0] = rng.normal(size=(S, A))   # varies with the action
phi[:, :, 1] = rng.normal(size=(S, 1))   # same for every action in a state

print({k: round(v, 2) if isinstance(v, float) else v
       for k, v in feature_diagnostics(phi).items()})
```

```text
{'num_features': 2, 'feature_rank': 2, 'condition_number': 1.29,
 'contrast_rank': 1, 'contrast_condition_number': 1.0}
```

The design has full rank, but `contrast_rank` is 1, not 2. The second feature is
constant across actions, so one parameter cannot be recovered. When
`contrast_rank` equals `num_features`, every parameter is identified from
behavior.

## Fit and read the outputs

```python
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
print(model.params_)
```

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
```

A fitted estimator exposes a common set of attributes.

| Attribute | What it holds |
| --- | --- |
| `params_` | Estimated reward or cost parameters. |
| `policy_` | Estimated choice probabilities, one row per state. |
| `value_` | Estimated value function, when the estimator computes one. |
| `summary()` | A report of the estimates with standard errors. |

## One counterfactual

Change a parameter and read the new policy.

```python
cf = model.counterfactual(RC=4.0)
print(cf.params)
print("replace probability at state 50:", float(np.asarray(cf.policy)[50, 1]))
```

```text
{'theta_c': 0.0010028828858836278, 'RC': 4.0}
replace probability at state 50: 0.05519477716656161
```

A higher replacement cost lowers the chance of replacing at a given mileage.

## Next steps

- Pick an estimator for your problem in the [estimator map](../estimators.md).
- Read the [overview](overview.md) for the ideas behind these models.
- Look up any class in the [API reference](../api/index.rst).
