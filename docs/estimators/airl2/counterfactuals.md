# Counterfactuals

## Important Links

- [AIRL2](../airl2.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Serialized-Content Example](serialized_content.md)

Read this page as anchored segment-level re-solving. Counterfactuals are
interpretable only when the exit-action and absorbing-state normalizations are
kept fixed under the intervention.

AIRL2 recovers a segment-level reward with the same parameterization as the
structural truth, provided the anchor normalization is correctly specified. Counterfactual
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

seg_rewards = model.segment_reward_matrices_
operator = SoftBellmanOperator(problem, transitions)

# Type A counterfactual: shift reward for segment k
for k, rw in enumerate(seg_rewards):
    rw_cf = jnp.array(rw).at[:, 0].add(0.5)
    rw_cf = rw_cf.at[:, model.exit_action].set(0.0)
    rw_cf = rw_cf.at[model.absorbing_state, :].set(0.0)
    result = value_iteration(operator, rw_cf)
    print(f"segment {k} counterfactual policy shape:", result.policy.shape)
```

**Result**

```text
segment 0 counterfactual policy shape: (21, 3)
segment 1 counterfactual policy shape: (21, 3)
```

## Counterfactual Families

| Family | Intervention | Checked against |
| --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | Oracle segment policy, value, and welfare regret. |
| Type B | Transition change (the dynamics change). | Oracle segment policy, value, and welfare regret. |
| Type C | Action removal (one action is penalized away). | Oracle segment policy, value, and welfare regret. |

## Reported Results

On the primary synthetic cell, welfare regret is reported per segment and the
maximum across segments is the conservative summary. Results from
[airl2.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl2.json):

| Counterfactual | Max regret across segments |
| --- | ---: |
| Type A | 0.0145 |
| Type B | 0.1189 |
| Type C | 0.00687 |

In this synthetic cell, Type B has the largest regret. It requires the
recovered reward to transfer to different dynamics without a likelihood-based
correction.

## Anchor Requirement

The anchor normalization is necessary for counterfactual validity. Without it,
the recovered reward contains potential-based perturbations that cancel in the
base world but produce different values under counterfactual dynamics or reward
shifts. The exit-action and absorbing-state anchors remove those perturbations,
so the counterfactual solve produces the correct structural response.
