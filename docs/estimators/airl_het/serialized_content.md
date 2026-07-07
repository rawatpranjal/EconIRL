# Serialized-Content Example

Read this page as the concrete AIRL-Het environment behind the validation
claim. The example shows why exit and absorbing-state anchors are part of the
model, not just implementation details.

The primary validation environment for AIRL-Het is a synthetic
serialized-content dynamic choice problem. Individuals decide each period
whether to read (pay and advance), wait (hold position at no cost), or exit
(leave the series permanently). The exit action absorbs the individual into a
terminal state. Two latent segments differ in their sensitivity to quality
versus price.

This page shows how to run AIRL-Het end-to-end on a small version of that
environment. The showcase script at
`examples/serialized-content/airl_het_showcase.py` gives a self-contained
runnable example with its own DGP.

```python
from econirl.estimation.adversarial.airl_het import AIRLHetConfig, AIRLHetEstimator
import jax.numpy as jnp

# Assume panel, utility, problem, transitions are already built.
# exit_action = 2 (exit), absorbing_state = n_episodes (terminal state).

config = AIRLHetConfig(
    num_segments=2,
    exit_action=2,
    absorbing_state=20,
    reward_type="linear",
    initialization="behavioral_anchor",
    max_em_iterations=30,
    verbose=True,
)

estimator = AIRLHetEstimator(config=config)
summary = estimator.estimate(panel, utility, problem, transitions)

# Segment priors and assignment
print(summary.metadata["segment_priors"])
print(summary.metadata["segment_assignments"][:10])

# Segment-specific policy (action probabilities per state)
for k, pol in enumerate(summary.metadata["segment_policies"]):
    print(f"segment {k} policy shape:", jnp.array(pol).shape)

# Segment-specific reward matrices
for k, rw in enumerate(summary.metadata["segment_reward_matrices"]):
    print(f"segment {k} reward shape:", jnp.array(rw).shape)
```

## Interpreting Segment Outputs

The estimated segments may not align with the data-generating segment labels in
an arbitrary order. The simulation study runner finds the minimum-cost permutation
between estimated and true segments before reporting accuracy and NRMSE
metrics. When comparing against ground truth outside the simulation study, align
segments manually by the prior or by reward similarity before computing
any per-segment metric.

## Replication Boundary

This page is an illustration of the package API on the serialized-content
environment, not a full replication of the LSW (2026) empirical results. The
estimator's recovery properties are established on the synthetic cell whose
data-generating process is fully specified; see the
[Simulation Study](validation.md) page. The empirical serialized-content
analysis requires proprietary consumption data and is outside this package
repository.
