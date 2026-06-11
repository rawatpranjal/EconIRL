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

The public docs describe the supported estimator surface and the current
validation scope for each method.

| Estimator | Best for | Public evidence scope |
| --- | --- | --- |
| [NFXP](https://econirl.readthedocs.io/en/latest/estimators/nfxp.html) | Exact tabular dynamic discrete choice. | Synthetic tabular validation. |
| [CCP / NPL](https://econirl.readthedocs.io/en/latest/estimators/ccp.html) | Hotz-Miller and NPL-style tabular DDC. | Synthetic tabular validation with support conditions. |
| [MPEC](https://econirl.readthedocs.io/en/latest/estimators/mpec.html) | Constrained likelihood formulation. | Synthetic constrained-likelihood validation. |
| [NNES](https://econirl.readthedocs.io/en/latest/estimators/nnes.html) | Neural value approximation inside NPL. | Low- and high-dimensional synthetic validation. |
| [TD-CCP](https://econirl.readthedocs.io/en/latest/estimators/tdccp.html) | Transition-density-free CCP estimation. | Encoded-state finite-theta validation. |
| [MCE-IRL](https://econirl.readthedocs.io/en/latest/estimators/mce_irl.html) | Maximum causal entropy reward-feature matching. | Supplied-feature simulation validation. |
| [Deep MCE-IRL](https://econirl.readthedocs.io/en/latest/estimators/deep_mce_irl.html) | Neural reward-map recovery with known transitions. | Anchored neural reward-map validation. |
| [AIRL](https://econirl.readthedocs.io/en/latest/estimators/airl.html) | Adversarial state-reward recovery under AIRL assumptions. | State-only AIRL validation. |
| [AIRL-Het](https://econirl.readthedocs.io/en/latest/estimators/airl_het.html) | Anchored adversarial recovery with latent segments. | Serialized-content simulation validation. |
| [f-IRL](https://econirl.readthedocs.io/en/latest/estimators/f_irl.html) | f-divergence state-marginal matching. | State-marginal simulation validation. |
| [GLADIUS](https://econirl.readthedocs.io/en/latest/estimators/gladius.html) | Neural Q and continuation modeling. | Preview diagnostics. |
| [IQ-Learn](https://econirl.readthedocs.io/en/latest/estimators/iq_learn.html) | Inverse soft-Q learning. | Preview diagnostics. |

## Package Surface

The recommended API is sklearn-style:

```python
from econirl import NFXP, CCP, MPEC, NNES, TDCCP, MCEIRL
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
