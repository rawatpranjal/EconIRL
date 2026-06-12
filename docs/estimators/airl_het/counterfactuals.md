# Counterfactuals

AIRL-Het recovers a segment-level reward in the same gauge as the structural
truth, provided the anchor normalization is correctly specified. Counterfactual
analysis reruns each segment's dynamic program under a controlled change and
reads off the new segment policy and value. Segment-level counterfactuals can
diverge: an intervention that raises the value of the read action may
strengthen the high-engagement segment's behavior while having little effect on
the low-engagement segment.

The estimator does not expose a one-call `counterfactual` method. Use the
segment-specific reward matrices from the metadata together with the package's
solver utilities to re-solve under each intervention.

```python
from econirl.core.solvers import value_iteration
from econirl.core.bellman import SoftBellmanOperator
import jax.numpy as jnp

seg_rewards = summary.metadata["segment_reward_matrices"]  # list of K reward matrices
operator = SoftBellmanOperator(problem, transitions)

# Type A counterfactual: shift reward for segment k
for k, rw in enumerate(seg_rewards):
    rw_cf = jnp.array(rw).at[:, 0].add(0.5)  # raise read-action reward
    rw_cf = rw_cf.at[:, exit_action].set(0.0) # enforce anchor
    result = value_iteration(operator, rw_cf)
    print(f"segment {k} counterfactual policy TV:", ...)
```

## Counterfactual Families

| Family | Intervention | Checked against |
| --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | Oracle segment policy, value, and welfare regret. |
| Type B | Transition change (the dynamics change). | Oracle segment policy, value, and welfare regret. |
| Type C | Action removal (one action is penalized away). | Oracle segment policy, value, and welfare regret. |

## Reported Results

On the primary synthetic cell, welfare regret is reported per segment and the
maximum across segments is used for the gate. Results from
[aairl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json):

| Counterfactual | Max regret across segments | Threshold | Status |
| --- | ---: | ---: | --- |
| Type A | 0.0145 | 0.12 | pass |
| Type B | 0.1189 | 0.12 | pass |
| Type C | 0.00687 | 0.12 | pass |

Type B regret is close to its threshold. The transition-change counterfactual
is more demanding for adversarial estimators than reward shifts or action
removals because the reward must transfer across a dynamically different world
without a likelihood-based correction. The result still passes.

## Anchor Requirement

The anchor normalization is necessary for counterfactual validity. Without it,
the recovered reward contains potential-based perturbations that cancel in the
base world but produce different values under counterfactual dynamics or reward
shifts. The exit-action and absorbing-state anchors remove those perturbations,
so the counterfactual solve produces the correct structural response.
