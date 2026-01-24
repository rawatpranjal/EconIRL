# econirl scikit-learn Style API Design

> **Status:** Design document (loose rubric)
> **Date:** 2026-01-23

## Design Philosophy

Follow scikit-learn / DoubleML / EconML conventions:
- **Estimator(config).fit(data) → results as properties**
- Simple, composable, Jupyter-friendly
- No "Study", "Workflow", or "Replication" abstractions
- The estimator is the main object

---

## Core Estimator API

```python
from econirl import NFXP, CCP, NPL

est = NFXP(n_states=90, discount=0.9999)
est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

est.params_           # {'theta_c': 0.00107, 'RC': 9.35}
est.se_               # {'theta_c': 0.00012, 'RC': 1.23}
est.summary()         # Nice table
```

### Estimator Classes

```python
class NFXP:
    """NFXP estimator for dynamic discrete choice models."""

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | Callable = "linear_cost",
        se_method: str = "robust",  # "asymptotic", "robust", "bootstrap"
    ): ...

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        transitions: np.ndarray | None = None,
    ) -> "NFXP":
        """Fit model. Returns self for chaining."""
        ...

    # Fitted attributes (trailing underscore = sklearn convention)
    params_: dict[str, float]
    se_: dict[str, float]
    coef_: np.ndarray          # For sklearn compatibility
    log_likelihood_: float
    value_function_: np.ndarray
    transitions_: np.ndarray   # Estimated or provided

    # Output methods
    def summary(self) -> str: ...
    def to_latex(self) -> str: ...

    # Prediction methods
    def predict_proba(self, states: np.ndarray) -> np.ndarray: ...
    def simulate(self, n_agents: int, n_periods: int, seed=None) -> pd.DataFrame: ...
    def counterfactual(self, **params) -> CounterfactualResult: ...
```

### All Estimators Share Same Interface

```python
est_nfxp = NFXP(n_states=90, discount=0.9999).fit(df, ...)
est_ccp  = CCP(n_states=90, discount=0.9999).fit(df, ...)
est_npl  = NPL(n_states=90, discount=0.9999, n_iterations=5).fit(df, ...)

# Compare results
for est in [est_nfxp, est_ccp, est_npl]:
    print(est.__class__.__name__, est.params_)
```

---

## Data Handling

### Passing Data

Explicit column names (simple, clear):

```python
est.fit(df, state="mileage_bin", action="replaced", id="bus_id")
```

### State Construction

User handles preprocessing (transparent):

```python
df["mileage_bin"] = (df["mileage"] / 5000).astype(int).clip(0, 89)
est.fit(df, state="mileage_bin", ...)
```

Or use helper:

```python
from econirl.preprocessing import discretize_state
df["mileage_bin"] = discretize_state(df["mileage"], bins=90, bin_width=5000)
```

### Panel Validation

Fit internally validates:
- No missing values in key columns
- Actions are valid (0 to n_actions-1)
- States are valid (0 to n_states-1)
- Handles balanced/unbalanced panels

### Built-in Datasets

```python
from econirl.datasets import load_rust_bus

df = load_rust_bus()  # Ready-to-use DataFrame
# Columns: bus_id, period, mileage, mileage_bin, replaced
```

---

## Utility Specification

Three ways, from simple to flexible:

### 1. Built-in String

```python
est = NFXP(utility="linear_cost")  # u = -θc·s·(1-a) - RC·a
```

### 2. Callable

```python
def my_utility(state, action, params):
    theta_c, RC = params[0], params[1]
    return -theta_c * state * (1 - action) - RC * action

est = NFXP(utility=my_utility, n_params=2)
```

### 3. Utility Class (Full Control)

```python
from econirl import Utility

class QuadraticCost(Utility):
    n_params = 3
    param_names = ["theta_1", "theta_2", "RC"]
    param_bounds = [(0, None), (0, None), (0, None)]
    param_init = [0.001, 0.0001, 10.0]

    def __call__(self, state, action, params):
        cost = params[0] * state + params[1] * state**2
        return -cost * (1 - action) - params[2] * action

est = NFXP(utility=QuadraticCost())
```

### Built-in Utilities

| Name | Formula | Params |
|------|---------|--------|
| `"linear_cost"` | -θc·s·(1-a) - RC·a | θc, RC |
| `"quadratic_cost"` | -(θ1·s + θ2·s²)·(1-a) - RC·a | θ1, θ2, RC |
| `"piecewise_linear"` | Piecewise by state bins | θ1...θk, RC |

---

## Transitions (First Stage)

### Automatic (Default)

```python
est.fit(df, ...)
est.transitions_  # Estimated from data during fit
```

### Explicit First Stage

```python
from econirl import TransitionEstimator

trans = TransitionEstimator().fit(df, state="mileage_bin", id="bus_id")
trans.matrix_     # Transition probability matrix
trans.summary()   # First-stage results table

est.fit(df, ..., transitions=trans.matrix_)
```

---

## Counterfactuals

```python
# What if replacement cost was higher?
cf = est.counterfactual(RC=15.0)
cf.value_function    # New V(s)
cf.choice_probs      # New P(a|s)
cf.expected_cost     # Expected total cost

# Compare policies
cf.policy_difference(est)  # How choices change
```

---

## Output Methods

### Summary Table

```python
est.summary()
```

```
NFXP Estimation Results
=======================
Log-likelihood: -1234.56
Observations: 8260

Parameter    Estimate    Std.Err    95% CI
---------    --------    -------    ------
theta_c      0.00107     0.00012    [0.00084, 0.00130]
RC           9.35        1.23       [6.94, 11.76]
```

### LaTeX Export

```python
est.to_latex()  # Returns string
est.to_latex(path="tables/results.tex")  # Saves to file
```

### Simulation

```python
sim_df = est.simulate(n_agents=100, n_periods=50, seed=42)
# Returns DataFrame: agent_id, period, state, action, cumulative_cost
```

---

## Full Jupyter Example

```python
import pandas as pd
from econirl import NFXP
from econirl.datasets import load_rust_bus

# Load data
df = load_rust_bus()

# Fit model
est = NFXP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    se_method="robust",
)
est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

# View results
print(est.summary())

# Counterfactual: 50% higher replacement cost
cf = est.counterfactual(RC=est.params_["RC"] * 1.5)
print(f"Expected cost increase: {cf.expected_cost / est.expected_cost_:.1%}")

# Simulate new data
sim = est.simulate(n_agents=1000, n_periods=100)
print(sim.groupby("period")["action"].mean())  # Replacement rate over time
```

---

## Implementation Roadmap

### Phase 1: Core Refactor
- [ ] Refactor `NFXPEstimator` → `NFXP` with sklearn API
- [ ] Add `params_`, `se_`, `coef_` properties
- [ ] Add `summary()` method
- [ ] DataFrame input with column names

### Phase 2: Other Estimators
- [ ] Refactor `CCPEstimator` → `CCP`
- [ ] Refactor `NPLEstimator` → `NPL`
- [ ] Ensure consistent interface across all

### Phase 3: Utilities
- [ ] Create `Utility` base class
- [ ] Implement built-in utilities
- [ ] Support callable utilities

### Phase 4: Extras
- [ ] `TransitionEstimator` class
- [ ] `counterfactual()` method
- [ ] `simulate()` method
- [ ] `to_latex()` export

### Phase 5: Polish
- [ ] Comprehensive docstrings
- [ ] Jupyter notebook examples
- [ ] API documentation

---

## Design Principles (Checklist)

- [ ] Estimator is the main object (no Study/Workflow wrappers)
- [ ] Configuration in `__init__`, data in `fit()`
- [ ] Results as properties with trailing underscore
- [ ] `fit()` returns self for chaining
- [ ] All estimators share same interface
- [ ] Works naturally in Jupyter
- [ ] Explicit column names, not magic
- [ ] Transitions automatic by default, explicit when needed
- [ ] Built-in utilities for common cases, custom for flexibility
