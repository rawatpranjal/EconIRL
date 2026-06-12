# AIRL

Adversarial inverse reinforcement learning learns a reward function through an
adversarial discriminator whose structure separates a transferable reward
component from a potential-based shaping term. In this package, AIRL is the
tabular IRL estimator closest to the original Fu et al. (2018) identification
setting. Its validated cell is state-only; the action-dependent diagnostic fails
by design.

Start here when behavior recovery in a state-only reward setting is the target,
or when you want an adversarial comparison alongside MCE-IRL. For anchored
action-dependent reward recovery, use AIRL-Het.

## Source Papers

This page draws on {ref}`Fu, Luo, and Levine (2018) <fu-2018>` for adversarial
inverse reinforcement learning and the potential-based shaping disentanglement
result.

## Quick Decision

| Use AIRL when | Prefer another estimator when |
| --- | --- |
| Demonstrations come from a discrete dynamic decision problem. | The reward is action-dependent (use AIRL-Het or MCE-IRL). |
| The reward target is state-only and the DGP satisfies decomposability. | You need structural standard errors or a likelihood-based estimate. |
| Adversarial recovery is the research object. | Counterfactual analysis needs a reward in the same parameterization as the data-generating process (use NFXP or MCE-IRL). |
| Transitions are available for policy update and evaluation. | Thin state coverage makes adversarial training unstable. |

## Minimal Fit

```python
from econirl.estimation import AIRLConfig
from econirl.estimation.adversarial import AIRLEstimator

config = AIRLConfig(
    reward_type="linear",
    reward_arg="state",
    use_shaping=True,
    max_rounds=150,
)
estimator = AIRLEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)
```

Set `reward_arg="state_action"` and increase `max_rounds` and `reward_lr` when
the DGP puts its signal in action contrasts. The state-only default matches the
original paper's identification setting.

## Evidence

AIRL is reported on a state-only synthetic cell designed to match the original
AIRL identification conditions. A second action-dependent cell is also run and
fails every gate; it is a documented boundary case, not a validated use case.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation. |
| Primary cell | `airl_paper_identification` (state-only). |
| Machine-readable results file | [airl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl.json). |
| Behavioral checks | Reward NRMSE, policy TV, value NRMSE, Q NRMSE, Type A/B/C regret. |
| Diagnostic cell | `airl_anchor_action_dependent` fails all gates. |

## AIRL Guide

- [Context](airl/context.md)
- [Quick Start](airl/quick_start.md)
- [Under the Hood](airl/under_the_hood.md)
- [Pre-Estimation Checks](airl/pre_estimation.md)
- [Simulation Study](airl/validation.md)
- [Counterfactuals](airl/counterfactuals.md)
- [Identification Boundary](airl/identification.md)

```{toctree}
:hidden:

airl/context
airl/quick_start
airl/under_the_hood
airl/pre_estimation
airl/validation
airl/counterfactuals
airl/identification
```
