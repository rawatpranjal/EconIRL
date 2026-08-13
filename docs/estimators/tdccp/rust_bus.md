# Rust Bus Engine Example

## Important Links

- [TD-CCP overview](../tdccp.md)
- [Quick Start](quick_start.md)
- [Evidence](validation.md)
- [Bus engine simulation study](../../simulation_studies/rust_bus.md)

The bundled synthetic Rust-style panel is a compact API example. It is not the
original Rust sample or the paper replication. The exact paper comparison uses
1,000 panels regenerated from the official Adusumilli and Eckardt (2025) code
and seed schedule for Online Appendix Table E.1.

```python
import numpy as np

from econirl import TDCCP
from econirl.datasets import load_rust_bus, rust_bus_reward_spec

data = load_rust_bus()
model = TDCCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility=rust_bus_reward_spec(90),
    method="semigradient",
    seed=7,
    basis_dim=8,
    cross_fitting=False,
    robust_se=False,
    se_method="asymptotic",
    outer_max_iter=1000,
    outer_tol=1e-7,
)
model.fit(
    data,
    state="mileage_bin",
    action="replaced",
    id="bus_id",
)

print(data.shape)
print(model.converged_)
print(np.round(model.coef_, 4).tolist())
print(model.policy_.shape)
print(model.transition_source_)
```

**Result**

```text
(9410, 6)
True
[-0.0013, 2.922]
(90, 2)
estimated from fitted panel
```

The parameter names come from `rust_bus_reward_spec(90)`. The first coefficient
multiplies `-mileage` for keep. The second multiplies `-1` for replacement. The
negative operating-cost estimate makes keep utility increase with mileage.
Treat this run as an API example, not evidence of parameter recovery. The
wrapper estimates a transition tensor from the panel because none was supplied.
That tensor is used for the fitted policy and value, not for the TD parameter
equations.

This example turns off cross-fitting to keep the example fast. Use the default
`se_method="robust"`, `cross_fitting=True`, and `robust_se=True` for Algorithm 2
inference on an adequately supported panel.

## Paper Comparison

The official Table E.1 comparison is different from this bundled-data example.
It uses the paper's state design, sample size, fold assignment, initial values,
and 1,000 regenerated simulation panels. Both the plug-in and locally robust
columns are compared with the published means and empirical standard
deviations. See [Evidence](validation.md) for the result.
