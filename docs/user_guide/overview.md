# Overview

EconIRL estimates models where agents make repeated choices while anticipating
future consequences. It combines classical dynamic discrete choice estimators
with inverse reinforcement learning estimators behind one Python interface.

Use EconIRL when the question is structural rather than only predictive. The
outputs include reward objects, policies, value functions, and counterfactual
policy calculations where the estimator supports them.

## What are DDC/IRL models even about?

Dynamic discrete choice models study repeated decisions. A user, firm, driver,
or operator chooses an action today while anticipating how that choice changes
future states. The model connects observed choices to payoff tradeoffs.

Inverse reinforcement learning starts from a similar decision problem but asks
for the reward model behind observed behavior. Instead of only predicting the
next action, IRL estimates what the decision maker appears to value and how
behavior changes under new incentives or environments.

EconIRL brings these two traditions into one workflow. Fit an estimator,
inspect the fitted rewards and policies, then run supported counterfactuals.

## Installation

```bash
pip install econirl
```

## Quick Start

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NFXP

df = load_rust_bus()
model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
cf = model.counterfactual(replacement_cost=4.0)
print(cf.policy[50, 1])
```

Output

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
0.05519477716656161
```

## Where to Start

| Question | Documentation entry point |
| --- | --- |
| I have my own panel of decisions. | Read [using your own data](your_own_data.md). |
| I need a tabular DDC baseline. | Use [NFXP](../estimators/nfxp.md). |
| I need to compare methods. | Use the [estimator map](../estimators.md). |
| I need the signature of a class. | Read the [API reference](../api/index.rst). |
| I need to understand simulation-study evidence. | Read [simulation studies](../simulation_studies/index.md). |

## Latest Updates

The documentation separates available methods from simulation-study evidence.
Each estimator page states the target, the reported evidence, and the
conditions under which the method should be used.

NFXP is the reference structural estimator for tabular dynamic discrete choice.
