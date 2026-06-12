# Pre-Estimation Checks

AIRL-Het has a richer failure surface than single-segment estimators because
segment identification depends on both the anchor design and the behavioral
signal that separates segments. Check these before fitting:

| Check | Why it matters for AIRL-Het |
| --- | --- |
| Anchor validity | If the exit-action index or absorbing-state index is wrong, the anchor constraints enforce the wrong normalization and reward recovery fails. |
| Feature rank | A rank-deficient design matrix leaves reward directions unidentified even under correct anchors. |
| Feature condition number | Ill-conditioning inflates the adversarial gradient noise and slows convergence. |
| State coverage | Reward recovery depends on the discriminator seeing transitions from most states. The three unobserved states in the primary cell are a known boundary. |
| State-action coverage | Rare action-state pairs are weakly identified. The minimum action share measures exposure to rare actions. |
| Segment behavioral separation | EM identifies segments only if the segments choose differently across states. Very similar segments require more data to separate. |
| Within-individual trajectory count | The consistency constraint requires at least two trajectories per individual to be useful; single-trajectory users contribute to the prior but not to within-user smoothing. |
| Transition row sums | Transition tensors must be row-stochastic in the $(A, S, S)$ orientation. |

## Primary Simulation Checks

Values from the primary `airl_het_paper_identification` run recorded in
[aairl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json):

| Check | Value | Status |
| --- | --- | --- |
| Feature rank | 20 / 20 | pass |
| Feature condition number | 20.411 | pass |
| Observed states | 58 / 61 | context |
| State-action coverage | 0.934 | pass |
| Minimum action share | 0.204 | pass |
| Max transition row error | 0.0 | pass |
| Anchor valid | true | pass |

The three unobserved states are outside the simulation support by design and do
not indicate a data problem. They remain visible here because segment
heterogeneity claims are support-sensitive.

## Common Risk Patterns

**Wrong anchor indices.** `exit_action` and `absorbing_state` must be the
correct integer indices for the specific environment. There are no defaults;
the estimator raises a `ValueError` if either is missing.

**Segment collapse.** One segment can absorb most of the prior mass if the
true behavioral difference is small or the initialization is unlucky. Monitor
`segment_priors` during EM; a prior near zero for any segment signals collapse.
The `prior_min` and `prior_smoothing` settings resist collapse.

**EM stopping too early.** The primary cell converges in 2 EM iterations, but
more complex DGPs may require more. Inspect `em_log_likelihoods` in the result
metadata to confirm the LL has stabilized before treating the output as final.
