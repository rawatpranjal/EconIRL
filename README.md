# econirl

The StatsModels of IRL - A Python package bridging Structural Econometrics and Inverse Reinforcement Learning.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from econirl import NFXP
from econirl.datasets import load_rust_bus

# Load data
df = load_rust_bus()

# Fit model
est = NFXP(n_states=90, discount=0.9999)
est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

# View results
print(est.params_)   # {'theta_c': 0.00107, 'RC': 9.35}
print(est.summary())

# Simulate
sim = est.simulate(n_agents=100, n_periods=50)

# Counterfactual: what if RC doubled?
cf = est.counterfactual(RC=est.params_["RC"] * 2)
```

### Available Estimators

| Estimator | Description |
|-----------|-------------|
| `NFXP` | Nested Fixed Point (Rust 1987) |
| `CCP` | Conditional Choice Probability (Hotz-Miller) |

All estimators share the same interface:
- `est.fit(df, state=, action=, id=)` - fit model
- `est.params_` - parameter estimates
- `est.se_` - standard errors
- `est.summary()` - formatted results
- `est.simulate()` - simulate choices
- `est.counterfactual()` - policy analysis

## Features

- Economist-friendly API (utility, preferences, characteristics)
- StatsModels-style `summary()` output
- Multiple estimation methods (NFXP, CCP, with MaxEnt planned)
- Rich inference (standard errors, confidence intervals, hypothesis tests)
- Gymnasium-compatible environments
- Counterfactual analysis and visualization

## Replication Packages

### Rust (1987) - Bus Engine Replacement

Replicate the classic dynamic discrete choice paper:

```python
from econirl.replication.rust1987 import table_v_structural

# Reproduce Table V structural estimates
table = table_v_structural(groups=[4])
print(table)
```

See the [full replication notebook](examples/rust_1987_replication.ipynb) for details.

## License

MIT
