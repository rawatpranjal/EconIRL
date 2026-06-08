# Overview

EconIRL estimates models where agents make repeated choices while anticipating
future consequences. It combines classical dynamic discrete choice estimators
with inverse reinforcement learning estimators behind one Python interface.

Use EconIRL when the question is structural rather than only predictive. The
output is a reward model, an estimated policy, a value function, and
counterfactual behavior after payoffs or environments change.

## What are DDC/IRL models even about?

Dynamic discrete choice models study repeated decisions. A user, firm, driver,
or operator chooses an action today while anticipating how that choice changes
future states. The model connects observed choices to the payoff tradeoffs that
make those choices optimal.

Inverse reinforcement learning starts from a similar decision problem but asks
for the reward model behind observed behavior. Instead of only predicting the
next action, IRL estimates what the decision maker appears to value and how
behavior changes under new incentives or environments.

EconIRL brings these two traditions into one workflow. Fit an estimator, inspect
the recovered rewards and policies, then run counterfactuals on the fitted
model.

## Installation

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

Output

```text
{'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
0.05519477716656161
```

## Where to Start

| Question | Documentation entry point |
| --- | --- |
| I need a trusted tabular DDC baseline. | Use [NFXP](../estimators/nfxp.md). |
| I need to compare methods. | Use the [estimator map](../estimators.md). |
| I need to understand the API shape. | Read [API design](api_design.md). |
| I need to understand validation evidence. | Read [validation](validation.md). |

## Latest Updates

The documentation separates available methods from validated claims. Each
estimator page states the supported target, the evidence behind that target,
and the conditions under which the method should be used.

NFXP is the reference structural estimator. It is the first page a new user
should read for a tabular dynamic discrete choice workflow.
