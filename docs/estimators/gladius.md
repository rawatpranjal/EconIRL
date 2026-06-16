# GLADIUS

GLADIUS learns Q and continuation-value functions from observed transitions
using neural networks, then projects the implied reward back onto a small set
of structural features. It is designed for dynamic discrete choice settings
where the state space is too large for tabular value methods. Start here when
you have high-dimensional state features, an anchor action with known rewards,
and want projected structural reward estimates without solving a full dynamic
program inside a parameter search.

## Source Papers

This page draws on {ref}`Kang, Yoganarasimhan, and Jain (2025) <kang-2025>`
for the offline IRL-DDC objective, the Q and continuation-value architecture,
and the anchor-action identification strategy.

## Quick Decision

| Use GLADIUS when | Prefer another estimator when |
| --- | --- |
| State features are high-dimensional and tabular value methods are costly. | The state space is small and tabular (use NFXP, CCP, or UFXP). |
| You have an anchor action with known rewards for identification. | There is no credible anchor action. |
| Projected structural reward recovery is the target. | Full structural counterfactual validity is required (GLADIUS has not reached this; see [Simulation Study](gladius/validation.md)). |
| A policy-imitation and projected-reward diagnostic is sufficient. | Structural standard errors are required (use the structural family). |

## Quick Start

```python
from econirl import GLADIUS

model = GLADIUS(
    n_actions=3,
    discount=0.95,
    anchor_action=2,
    max_epochs=300,
)
model.fit(df, state="state_bin", action="action", id="individual_id",
          features=reward_spec)

print(model.params_)     # projected structural parameters
print(model.policy_)     # imitation policy, shape (n_states, n_actions)
```

The lower-level `GLADIUSEstimator` gives direct control over the anchor reward
vector and Bellman penalty weight:

```python
from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

config = GLADIUSConfig(
    anchor_action=2,
    anchor_rewards=anchor_rewards,
    anchor_bellman_loss=True,
)
result = GLADIUSEstimator(config=config).estimate(
    panel, utility, problem, transitions
)
```

## Evidence

GLADIUS is reported on two related synthetic cells with known rewards,
transitions, policies, and counterfactual oracle objects. Both cells pair a
21-state regular structure with high-dimensional encoded state features and a
low-dimensional reward basis, so every recovery claim can be compared against
the truth. The current results show strong policy imitation and projected-reward
recovery, but raw Bellman reward and value recovery do not pass their checks.
GLADIUS is therefore reported as diagnostically useful, not as
counterfactual-valid.

| Evidence | Current state |
| --- | --- |
| Scope | Synthetic high-dimensional-state simulation. |
| Primary cell | `gladius_paper_high_state` (21 states, 64 state features, 3 actions). |
| Scaled cell | `gladius_paper_high_state_scaled` (21 states, 128 state features, 3 actions). |
| Results file (primary) | [gladius.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json). |
| Results file (scaled) | [gladius_scaled.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_scaled.json). |
| Counterfactual checks | Type A, B, and C regret reported; full structural certification not reached. |
| Public example | Uses `GLADIUS` (sklearn API) and `GLADIUSEstimator` (lower level). |

## GLADIUS Guide

- [Context](gladius/context.md)
- [Quick Start](gladius/quick_start.md)
- [Under the Hood](gladius/under_the_hood.md)
- [Pre-Estimation Checks](gladius/pre_estimation.md)
- [Simulation Study](gladius/validation.md)
- [Counterfactuals](gladius/counterfactuals.md)
- [High-State Example](gladius/high_state_example.md)

```{toctree}
:hidden:

gladius/context
gladius/quick_start
gladius/under_the_hood
gladius/pre_estimation
gladius/validation
gladius/counterfactuals
gladius/high_state_example
```
