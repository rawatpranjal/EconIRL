# Counterfactuals

Read this page as a diagnostic boundary page. IQ-Learn can produce low regret
while still failing reward, value, or Q recovery checks.

IQ-Learn recovers a Bellman-implied reward as a diagnostic alongside the
imitation policy. Using that reward for counterfactual analysis requires
understanding what is and is not grounded by the current evidence.

## What the Evidence Says

On the primary synthetic cell (`canonical_low_action`) the three counterfactual
regret checks pass:

| Family | Intervention | Regret | Gate |
| --- | --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | 0.011518059257921435 | pass |
| Type B | Transition change (the dynamics change). | 0.02155897051624924 | pass |
| Type C | Action removal (one action is penalized away). | 0.009886935172236239 | pass |

Low regret on this cell means the Q-induced policy happens to produce
near-oracle welfare under the applied interventions. It does not mean the
reward, value, or Q objects are structurally accurate; all three fail their
NRMSE checks by a wide margin on this same cell.

On the high-dimensional neural cell, Type A and Type B regret also fail.

## How to Use the Reward Diagnostic

The Bellman-implied reward `r_IB(s, a) = Q(s, a) - beta * E[V(s') | s, a]`
is available from `summary.metadata["raw_bellman_reward_table"]`. A
least-squares projection of this table into the utility feature basis is
stored in `summary.metadata["projected_reward_matrix"]` and
`summary.metadata["reward_params"]`.

These are diagnostic objects. They may be useful for qualitative comparison
or for initializing a structural estimator, but they should not be reported as
structurally recovered parameters without first verifying that the structural
checks are satisfied.

## Structural Counterfactual Use

To run a counterfactual with structural guarantees, use the structural family
(NFXP, CCP, MPEC, or UFXP). Those estimators enforce the Bellman fixed point
as a hard constraint and report standard errors; IQ-Learn does neither.

If you want to use the IQ-Learn projected theta as a starting point for a
structural estimator:

```python
# Extract projected parameters from IQ-Learn
iq_summary = estimator.estimate(panel, utility, problem, transitions)
theta_init = iq_summary.metadata.get("reward_params")

# Pass as starting point to NFXP or another structural estimator
from econirl.estimation import NFXPEstimator
nfxp_summary = NFXPEstimator().estimate(
    panel, utility, problem, transitions,
    initial_params=theta_init,
)
```

Check coverage gates before relying on `theta_init`; off-support reward values
can mislead the structural search.
