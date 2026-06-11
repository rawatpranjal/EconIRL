# AIRL

## Overview

AIRL learns a reward through an adversarial discriminator with a
potential-based shaping term. The current evidence follows the original
state-only AIRL identification setting.

Use AIRL-Het, not plain AIRL, for the anchored heterogeneous action-dependent
surface.

## Source Papers

This page draws on {ref}`Fu, Luo, and Levine (2018) <fu-2018>` for adversarial
inverse reinforcement learning.

## When to Use

Use AIRL when:

- demonstrations come from a discrete dynamic decision problem;
- the reward target is state-only;
- transitions are available for validation or post-fit evaluation;
- adversarial recovery is the research object.

Avoid plain AIRL when the reward is action-dependent, the absorbing-state gauge
is central, or the project requires action-dependent structural reward
recovery.

## Basic Usage

```python
from econirl.estimation import AIRLConfig
from econirl.estimation.adversarial import AIRLEstimator

config = AIRLConfig(
    reward_type="linear",
    reward_arg="state",
    use_shaping=True,
)
estimator = AIRLEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
```

The high-level `econirl.estimators.AIRL` alias points to the neural wrapper. The
tabular primer uses `AIRLEstimator`.

## Evidence

AIRL is reported on the original AIRL identification DGP.

The original AIRL identification DGP is a state-only reward setting close to
the paper's assumptions. Anchored action-dependent reward recovery belongs to
AIRL-Het in the current documentation.

## Further Reading

- Machine-readable results file: [airl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl.json)
