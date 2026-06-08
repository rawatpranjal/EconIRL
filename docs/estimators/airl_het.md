# AIRL-Het

## Overview

AIRL-Het extends AIRL to latent heterogeneous segments with segment-specific
rewards and policies. It uses anchor restrictions to identify
action-dependent rewards and an EM loop to recover segment membership.

The package validation is tied to a serialized-content DGP with repeated
observations per user, not to a generic one-trajectory-per-agent IRL problem.

## When to Use

Use AIRL-Het when:

- the population has latent segments with different dynamic preferences;
- users have repeated trajectories that help identify segment membership;
- an anchor action and absorbing-state normalization are credible;
- action-dependent reward recovery is the target;
- segment-specific counterfactuals matter.

Avoid AIRL-Het when segment membership is weakly identified, there is no
credible reward anchor, or a homogeneous estimator is enough.

## Basic Usage

```python
from econirl.estimation.adversarial import AIRLHetConfig, AIRLHetEstimator

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
```

Use user-level trajectory metadata consistently when repeated trajectories
belong to the same individual.

## Validation Status

AIRL-Het passes the known-truth gates on the serialized-content heterogeneous
DGP. The validated cell has two latent segments, repeated books per user,
pay/wait/exit actions, an exit-action reward anchor, an absorbing terminal
state, an 18-dimensional state encoding, and 20 finite reward features.

Serialized-content heterogeneous DGP means a known-truth dynamic choice problem
with repeated observations per user and segment-level reward, policy, value,
and counterfactual truth.

## Further Reading

- Machine-readable artifact: [aairl_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/aairl/aairl_results.json)
