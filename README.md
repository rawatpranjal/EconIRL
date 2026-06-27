# econirl

Structural dynamic discrete choice and inverse reinforcement learning in
Python.

EconIRL helps estimate forward-looking choice models, recover reward functions,
and evaluate counterfactual policies from panel data.

**Documentation:** https://econirl.readthedocs.io/

## Install

```bash
pip install econirl
```

## Quick Start

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

Example output:

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
0.05519477716656161
```

## Public Estimator Guides

The public docs center on one reference estimator, the nested fixed point (NFXP),
the exact estimator we replicate to Rust (1987) Table IX. Every other estimator
answers a specific complication that breaks NFXP's canonical case. See
[Choosing an Estimator](https://econirl.readthedocs.io/en/latest/estimators/landscape.html)
for the full map.

| Estimator | Best for | Public evidence scope |
| --- | --- | --- |
| [NFXP](https://econirl.readthedocs.io/en/latest/estimators/nfxp.html) | Exact tabular dynamic discrete choice, replicated to Rust (1987) Table IX. | Synthetic tabular validation and the Rust bus replication. |

The other estimators are grouped by the source of complexity they answer: large or
continuous state spaces (CCP, MPEC, UFXP, NNES), hard-to-model transitions
(TD-CCP), unknown reward form (MCE-IRL, Neural MCE-IRL, AIRL, GLADIUS), latent
heterogeneity (AIRL-Het), and finite-horizon planning (RHIP), with f-IRL, IQ-Learn,
and the contrib baselines alongside.

## Package Surface

The recommended API is sklearn-style:

```python
from econirl import NFXP, CCP, NNES, TDCCP, MCEIRL, RHIP, AIRL, GLADIUS
```

Additional estimators and lower-level configuration objects are available under
`econirl.estimation`, `econirl.estimators`, and `econirl.contrib` for advanced
workflows.

## Repository Layout

- `src/econirl/`: package source.
- `tests/`: unit, integration, and validation-evidence tests.
- `docs/`: public Read the Docs source.
- `validation/`: reproducible validation runners and machine-readable results.
- `examples/`: public examples and notebooks.

Manuscripts, PDFs, local research workspaces, and assistant-specific notes are
not tracked in this public package repository.

## License

MIT
