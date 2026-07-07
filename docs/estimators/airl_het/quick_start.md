# Quick Start

This page shows the anchored heterogeneous AIRL path. The exit action and
absorbing state are not optional bookkeeping; they are the normalizations that
make the segment rewards interpretable.

AIRL-Het is accessed through `AIRLHetConfig` and `AIRLHetEstimator`. Both the
exit action index and the absorbing state index must be supplied; the estimator
raises an error if either is missing.

```python
from econirl.estimation.adversarial.airl_het import AIRLHetConfig, AIRLHetEstimator

config = AIRLHetConfig(
    num_segments=2,
    exit_action=2,       # index of the anchor action (reward = 0)
    absorbing_state=20,  # index of the absorbing state (V = 0)
    reward_type="linear",
    initialization="behavioral_anchor",
    max_em_iterations=30,
    verbose=True,
)

estimator = AIRLHetEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)
```

Key outputs from the returned `EstimationSummary`:

| Attribute | Meaning |
| --- | --- |
| `summary.parameters` | Concatenated segment reward parameters (all segments, flattened). |
| `summary.policy` | Mixture policy (prior-weighted average over segments). |
| `summary.value_function` | Mixture value function. |
| `summary.converged` | Whether the EM loop converged by the tolerance criterion. |
| `summary.num_iterations` | Number of EM iterations completed. |
| `summary.log_likelihood` | Mixture log-likelihood at the final EM iterate. |
| `summary.metadata["segment_priors"]` | Estimated prior probability per segment. |
| `summary.metadata["segment_posteriors"]` | Posterior assignment probability per trajectory. |
| `summary.metadata["segment_policies"]` | Segment-specific policy arrays. |
| `summary.metadata["segment_reward_matrices"]` | Segment-specific reward matrices. |
| `summary.metadata["segment_assignments"]` | Hard segment assignment (argmax of posteriors). |

## Initialization

Two initialization schemes are available:

```python
config = AIRLHetConfig(..., initialization="random")             # random reward start
config = AIRLHetConfig(..., initialization="behavioral_anchor")  # cluster by behavior, invert anchor
```

`behavioral_anchor` clusters trajectories by observed action shares, inverts
the anchored soft policy into a starting reward for each cluster, and sets
the initial posteriors accordingly. It typically converges faster than a random
start for the serialized-content setting.

## Reward Type

```python
config = AIRLHetConfig(..., reward_type="tabular")   # state-action table (S x A parameters per segment)
config = AIRLHetConfig(..., reward_type="linear")     # linear features (one theta per segment)
```

The linear mode uses the feature matrix from the supplied utility object. The
tabular mode directly learns an entry per state-action pair and is more
flexible but requires a larger sample.
