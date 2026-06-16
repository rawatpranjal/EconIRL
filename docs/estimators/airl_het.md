# AIRL-Het

AIRL-Het extends adversarial inverse reinforcement learning to populations with
latent segments. Each segment has its own reward and policy, recovered jointly
through an EM loop that alternates between assigning trajectories to segments
and updating each segment's adversarial reward. Two anchor constraints, one on
the exit action and one on the absorbing state, pin down the action-dependent
reward uniquely so the recovered reward equals the structural one, not a shaped
perturbation.

Start here when the population is heterogeneous, individuals make repeated
decisions in the same environment, and action-dependent counterfactual reward
recovery is the target.

## Source Papers

This page draws on {ref}`Fu, Luo, and Levine (2018) <fu-2018>` for the base
AIRL construction and on
{ref}`Lee, Sudhir, and Wang (2026) <lee-sudhir-wang-2026>` for the anchored
heterogeneous extension and the serialized-content identification setting.

## Quick Decision

| Use AIRL-Het when | Prefer another estimator when |
| --- | --- |
| The population has latent segments with distinct dynamic preferences. | A single homogeneous reward is a defensible assumption. |
| Individuals have repeated trajectories that help identify segment membership. | Each individual appears only once or very briefly. |
| An exit action and an absorbing state are credibly available for anchor normalization. | No credible reward anchor exists. |
| Action-dependent structural reward recovery is the target. | State-only reward recovery is enough (use base AIRL). |
| Segment-specific counterfactuals are central. | Only fitted choice probabilities are needed. |

## Quick Start

```python
from econirl.estimation.adversarial.airl_het import AIRLHetConfig, AIRLHetEstimator

config = AIRLHetConfig(
    num_segments=2,
    exit_action=2,
    absorbing_state=20,
    reward_type="linear",
    initialization="behavioral_anchor",
)
estimator = AIRLHetEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.parameters)
print(summary.metadata["segment_priors"])
print(summary.metadata["segment_posteriors"])
```

Both `exit_action` and `absorbing_state` must be specified explicitly; they
have no sensible defaults.

## Evidence

AIRL-Het is evaluated on a synthetic serialized-content heterogeneous cell with
two latent segments, repeated observations per user, and three actions
(read/wait/exit). The cell has 61 states, 20 reward features, and known
segment-level rewards, policies, values, Q functions, and Type A, Type B, and
Type C counterfactual oracle objects.

| Evidence | Current state |
| --- | --- |
| Scope | Synthetic serialized-content simulation with two latent segments. |
| Primary cell | `airl_het_paper_identification`. |
| Results file | [aairl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json). |
| Counterfactual checks | Type A, Type B, and Type C reported per segment in the results file. |
| Coverage note | 58 of 61 states are observed; three states are outside the simulation support. |

## AIRL-Het Guide

- [Context](airl_het/context.md)
- [Quick Start](airl_het/quick_start.md)
- [Under the Hood](airl_het/under_the_hood.md)
- [Pre-Estimation Checks](airl_het/pre_estimation.md)
- [Simulation Study](airl_het/validation.md)
- [Counterfactuals](airl_het/counterfactuals.md)
- [Serialized-Content Example](airl_het/serialized_content.md)

```{toctree}
:hidden:

airl_het/context
airl_het/quick_start
airl_het/under_the_hood
airl_het/pre_estimation
airl_het/validation
airl_het/counterfactuals
airl_het/serialized_content
```
