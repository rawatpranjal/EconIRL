# GLADIUS

## Overview

GLADIUS learns Q and continuation-value approximations and projects the implied
reward back onto structural features. The reported evidence uses a stable
anchor-moment Q loss for synthetic cells. The literal bi-conjugate minimax
mode remains available for method research.

## When to Use

Use GLADIUS when:

- choices are dynamic and discrete;
- you have an anchor action with known rewards;
- high-dimensional state features make tabular value methods unattractive;
- projected structural reward recovery is the target.

Avoid GLADIUS when there is no credible anchor action or when you need tabular
structural estimation.

## Basic Usage

```python
from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

config = GLADIUSConfig(
    anchor_action=2,
    anchor_rewards=anchor_rewards,
    anchor_bellman_loss=True,
)
estimator = GLADIUSEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
```

The high-level `econirl.estimators.GLADIUS` alias points to the neural wrapper.
The primer evidence uses the lower-level `GLADIUSEstimator`.

## Evidence

GLADIUS is currently positioned as a research preview for projected reward
analysis. The reported evidence covers projected parameters, projected
rewards, policies, Q functions, and Type A/B/C counterfactual checks on the
paper-side high-dimensional-state cell and its scaled variant.

The high-dimensional-state GLADIUS DGP uses encoded state features,
low-dimensional reward features, known transitions, and an anchor action so the
projected structural reward can be checked against true simulated values.
The current results files still fail the raw Bellman reward and value-function
recovery gates. Passing projected-reward and counterfactual-regret checks is
therefore reported as a diagnostic result, not as full structural
counterfactual validity.

## Further Reading

- Machine-readable results file: [gladius_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/gladius/gladius_results.json)
- Scaled-cell results file: [gladius_scaled_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/gladius/gladius_scaled_results.json)
