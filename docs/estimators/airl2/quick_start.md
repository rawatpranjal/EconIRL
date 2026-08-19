# Quick Start

## Important Links

- [AIRL2](../airl2.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Simulation Study](validation.md)
- [Serialized-Content Example](serialized_content.md)

This page shows the anchored heterogeneous AIRL path. The exit action and
absorbing state are not optional bookkeeping; they are the normalizations that
make the segment rewards interpretable.

AIRL2 is imported from the package root. Both the exit action index and the
absorbing state index must be supplied. The constructor rejects either index
when it falls outside the declared state or action support.

```python
from econirl import AIRL2

model = AIRL2(
    n_states=21,
    n_actions=3,
    num_segments=2,
    exit_action=2,       # index of the anchor action (reward = 0)
    absorbing_state=20,  # absorbing reward and shaping anchor
    reward_type="linear",
    initialization="behavioral_anchor",
    max_em_iterations=30,
    compute_se=False,
    verbose=True,
)

print(model.num_segments)
print(model.n_states, model.n_actions)
```

**Result**

```text
2
21 3
```

After calling `fit`, key attributes on the model include:

| Attribute | Meaning |
| --- | --- |
| `model.params_` | Concatenated segment reward parameters. |
| `model.policy_` | Mixture policy, averaged with the estimated segment priors. |
| `model.value_` | Mixture value function. |
| `model.converged_` | Whether the EM loop met its tolerance criterion. |
| `model.n_iter_` | Number of EM iterations completed. |
| `model.log_likelihood_` | Mixture log-likelihood at the final EM iterate. |
| `model.segment_priors_` | Estimated prior probability per segment. |
| `model.segment_posteriors_` | Posterior assignment probability per trajectory. |
| `model.segment_policies_` | Segment-specific policy arrays. |
| `model.segment_reward_matrices_` | Segment-specific reward matrices. |
| `model.segment_assignments_` | Hard segment assignment from the posteriors. |

## Initialization

Two initialization schemes are available. Set `initialization="random"` for a
random reward start. Set `initialization="behavioral_anchor"` to cluster by
behavior and invert the anchor.

`behavioral_anchor` clusters trajectories by observed action shares, inverts
the anchored soft policy into a starting reward for each cluster, and sets
the initial posteriors accordingly. It typically converges faster than a random
start for the serialized-content setting.

## Reward Type

Set `reward_type="tabular"` for a state-action table with one entry per cell and
segment. Set `reward_type="linear"` for one feature coefficient vector per
segment.

The linear mode uses the feature matrix from the supplied utility object. The
tabular mode directly learns an entry per state-action pair and is more
flexible but requires a larger sample.

## Uncertainty

Set `compute_se=True` to run the individual-cluster bootstrap after the point
fit. Each resample keeps all trajectories from a sampled individual together.
AIRL2 aligns segment labels before forming pilot-calibrated normal intervals
for segment reward cells, policies, and prior shares. Use `model.conf_int()` to
retrieve the intervals.
