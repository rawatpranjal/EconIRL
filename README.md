# econirl

Structural dynamic discrete choice and inverse reinforcement learning in
Python.

EconIRL helps researchers and applied teams estimate forward-looking choice
models, recover interpretable reward functions, and evaluate policy
counterfactuals from panel data.

**Docs:** https://econirl.readthedocs.io/

## Install

```bash
pip install econirl
```

## Quick Start

Load a bundled Rust bus dataset and fit the structural reference estimator.

```python
from econirl.datasets import load_rust_bus
from econirl import NFXP

df = load_rust_bus()
model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
print(model.params_)
cf = model.counterfactual(RC=4.0)
print(cf.policy[50, 1])
```

Output

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
0.05519477716656161
```

## What Is Validated Now

The estimator pages state the supported target and link to validation evidence
where a machine-readable artifact is available.

| Estimator | Role | Current validation status |
| --- | --- | --- |
| [NFXP](https://econirl.readthedocs.io/en/latest/estimators/nfxp.html) | Exact nested fixed-point MLE | Validated reference estimator |
| [CCP / NPL](https://econirl.readthedocs.io/en/latest/estimators/ccp.html) | Hotz-Miller inversion with NPL updates | Validated with stated support conditions |
| [MPEC](https://econirl.readthedocs.io/en/latest/estimators/mpec.html) | Constrained likelihood formulation | Validated as an NFXP counterpart |
| [SEES](https://econirl.readthedocs.io/en/latest/estimators/sees.html) | Sieve value approximation | Validated with optimizer-scope notes |
| [NNES](https://econirl.readthedocs.io/en/latest/estimators/nnes.html) | Neural NPL value approximation | Validated on low- and high-dimensional cells |
| [MCE-IRL](https://econirl.readthedocs.io/en/latest/estimators/mce_irl.html) | Maximum causal entropy feature matching | Validated for supplied reward features |

## Estimator Families

EconIRL covers classical structural estimators and modern IRL estimators behind
a common workflow.

| Family | Estimators |
| --- | --- |
| Structural econometrics | NFXP, CCP / NPL, MPEC, SEES, NNES, TD-CCP |
| Entropy and feature-matching IRL | MCE-IRL, Deep MCE-IRL, MaxEnt IRL, Bayesian IRL |
| Margin and distribution matching | Max Margin IRL, Max Margin Planning, f-IRL |
| Neural / Q-based methods | GLADIUS, Neural GLADIUS, IQ-Learn |
| Adversarial IRL | AIRL, Neural AIRL, AIRL-Het, GAIL, GCL |
| Baselines and utilities | Behavioral cloning, transition estimation, Rust bus replication tools |

## Package Surface

The recommended public API is sklearn-style:

```python
from econirl import (
    NFXP,
    CCP,
    SEES,
    NNES,
    TDCCP,
    MCEIRL,
    MaxEntIRL,
    MaxMarginIRL,
    GLADIUS,
    NeuralGLADIUS,
    AIRL,
    NeuralAIRL,
    IQLearn,
    MCEIRLNeural,
)
```

Lower-level estimator implementations remain available under
`econirl.estimation` and `econirl.contrib` for advanced workflows that need
direct access to configuration objects, panel objects, transition tensors, or
research implementations.

## Reference Pages

- Estimator index: https://econirl.readthedocs.io/en/latest/estimators.html
- NFXP guide: https://econirl.readthedocs.io/en/latest/estimators/nfxp.html

## License

MIT
