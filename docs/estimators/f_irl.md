# f-IRL

## Overview

f-IRL learns a reward by matching expert and model visitation distributions
under an f-divergence. The paper-side claim is state-marginal matching with a
state-only reward.

Action-dependent DDC examples are comparison exercises in the current package.
The supported structural target is state-marginal reward recovery.

## When to Use

Use f-IRL when:

- the reward target is state-only;
- state marginal matching is the right validation surface;
- transitions are available for policy evaluation;
- you want an f-divergence IRL baseline rather than a likelihood-based DDC
  estimator.

Avoid f-IRL for generic action-dependent structural DDC reward recovery.

## Basic Usage

```python
from econirl.estimation import FIRLEstimator

estimator = FIRLEstimator(
    f_divergence="fkl",
    marginal_space="state",
    reward_scope="state",
)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
```

Use `marginal_space="state_action"` only when the project is explicitly about a
state-action comparison exercise.

## Validation Status

The current validation target is the paper-faithful state-marginal f-IRL DGP.
That is the structural claim supported by this page.

The state-marginal f-IRL DGP uses state-only rewards and state-marginal
matching.

## Further Reading

- Machine-readable artifact: [f_irl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/f_irl/f_irl_results.json)
