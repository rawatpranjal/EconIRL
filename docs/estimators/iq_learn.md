# IQ-Learn

IQ-Learn learns a soft Q-function from expert data. The Q-function implicitly
encodes both the policy and the Bellman-implied reward, so the training
objective collapses the usual adversarial min-max into a single concave
optimization over Q alone.

Start here when you want an imitation estimator that avoids adversarial
training and produces a Q-based reward diagnostic alongside the policy.

## Source Papers

This page draws on {ref}`Garg et al. (2021) <garg-2021>` for the inverse
soft-Q learning construction and divergence objectives.

## Quick Decision

| Use IQ-Learn when | Prefer another estimator when |
| --- | --- |
| You want a policy imitation estimator without adversarial training. | Structural counterfactual analysis is the primary goal (use NFXP, CCP, or UFXP). |
| A Q-based reward diagnostic is useful alongside the policy fit. | The data-generating process is linear and transitions are known (use UFXP or CCP). |
| You want to compare behavioral cloning with a Bellman-aware alternative. | Reward recovery with structural standard errors is required (use NFXP or MPEC). |
| Transitions are available to compute the inverse Bellman operator. | State or state-action coverage is very thin (Q and reward recovery degrade). |

## Quick Start

```python
from econirl.estimation import IQLearnConfig, IQLearnEstimator

config = IQLearnConfig(
    q_type="tabular",
    divergence="chi2",
    alpha=1.0,
)
estimator = IQLearnEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
```

Use `q_type="linear"` with a feature-based utility to let the reward propagate
to unvisited state-action pairs. Do not pair `q_type="tabular"` with
`divergence="simple"`: the simple objective is unbounded on a free Q table and
will drive Q to numerical overflow.

## Evidence

IQ-Learn is validated on three synthetic cells. On the primary cell
(`canonical_low_action`, 21 states, 3 actions, 160 000 observations) the
estimator converges, achieves full state and state-action coverage, and passes
policy TV and counterfactual regret checks. It does not pass raw Bellman
reward, projected reward, value, or Q recovery checks on any tested cell. Use
the output as an imitation and Q-diagnostic tool rather than a source of
structural counterfactual evidence.

| Evidence | Current state |
| --- | --- |
| Scope | Synthetic tabular simulation. |
| Primary cell | `canonical_low_action`. |
| Results file | [iq_learn.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/iq_learn.json). |
| Imitation and regret checks | Pass on the primary cell (policy TV, Type A, Type B, Type C regret). |
| Structural recovery checks | Fail on all tested cells (reward, value, Q NRMSE). |

## IQ-Learn Guide

- [Context](iq_learn/context.md)
- [Quick Start](iq_learn/quick_start.md)
- [Under the Hood](iq_learn/under_the_hood.md)
- [Pre-Estimation Checks](iq_learn/pre_estimation.md)
- [Simulation Study](iq_learn/validation.md)
- [Counterfactuals](iq_learn/counterfactuals.md)

```{toctree}
:hidden:

iq_learn/context
iq_learn/quick_start
iq_learn/under_the_hood
iq_learn/pre_estimation
iq_learn/validation
iq_learn/counterfactuals
```
