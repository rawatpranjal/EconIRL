# Rust Bus Engine Example

## Important Links

- [CCP Overview](../ccp.md)
- [Quick Start](quick_start.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)

Read this page as a runnable CCP smoke test, not as the main evidence page. It
shows the wrapper on a familiar problem after the estimator assumptions have
already been checked elsewhere.

The Rust bus-engine replacement problem is the canonical dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep the
current engine or replace it.

By default, `load_rust_bus()` returns a synthetic dataset with Rust-style
states and choices. It does not load the historical Rust sample.

```python
from econirl import CCP
from econirl.datasets import load_rust_bus, rust_bus_reward_spec

df = load_rust_bus()

model = CCP(
    n_states=90,
    discount=0.9999,
    utility=rust_bus_reward_spec(90),
    num_policy_iterations=3,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

for name in model.params_:
    print(f"{name}: estimate={model.params_[name]:.6f}, se={model.se_[name]:.6f}")
print(f"termination={model.termination_reason_}")
print(f"npl_converged={model.npl_converged_}")
print(f"parameter_residual={model.npl_parameter_residual_:.6e}")
print(f"policy_residual={model.npl_policy_residual_:.6e}")
```

**Result**

```text
operating_cost: estimate=0.000995, se=0.000421
replacement_cost: estimate=3.072211, se=0.074237
termination=fixed_k_complete
npl_converged=False
parameter_residual=1.452655e-02
policy_residual=2.148578e-02
```

The model completed all three requested NPL stages. The parameter and policy
residuals did not both meet the fixed-point tolerance within those stages.

## Interpretation

The Rust bus reward specification estimates two parameters. The first is the
operating cost slope over mileage states. The second is the replacement cost.
The fitted policy gives the replacement probability by mileage state.

```python
for state, row in zip(
    [0, 10, 50, 89],
    model.predict_proba([0, 10, 50, 89]),
    strict=True,
):
    print(f"{state}: keep={row[0]:.6f}, replace={row[1]:.6f}")
```

**Result**

```text
0: keep=0.955732, replace=0.044268
10: keep=0.947600, replace=0.052400
50: keep=0.912779, replace=0.087221
89: keep=0.884699, replace=0.115301
```

## Counterfactual Replacement Cost

```python
cf = model.counterfactual(replacement_cost=4.0)
print(f"P(replace | state=50)={cf.policy[50, 1]:.6f}")
```

**Result**

```text
P(replace | state=50)=0.054908
```

Raising replacement cost from the fitted value to 4.0 reduces the replacement
probability at state 50 from 0.087221 to 0.054908.

## Replication Boundary

This page is a package smoke example, not the full historical Rust
replication. The printed robust standard errors are conditional
pseudo-likelihood estimates. They hold the fitted transition model fixed. The
reported simulation study is also synthetic and supplies the true transition
tensor. The [replication ledger](../../replications.md) separates two results.
Its CCP/NPL section checks fixed-point equivalence to NFXP on bundled Group 4
data. Its Rust Table IX section uses NFXP and official data to reproduce the
published estimates and standard errors.
