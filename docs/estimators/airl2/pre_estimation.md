# Pre-Estimation Checks

## Important Links

- [AIRL2](../airl2.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Counterfactuals](counterfactuals.md)

Read this page before fitting AIRL2. The anchors and segment separation are
not secondary diagnostics; they are what make heterogeneous reward recovery
interpretable.

AIRL2 has a richer failure surface than single-segment estimators because
segment identification depends on both the anchor design and the behavioral
signal that separates segments. Check these before fitting:

| Check | Why it matters for AIRL2 |
| --- | --- |
| Anchor validity | If the exit-action index or absorbing-state index is wrong, the anchor constraints enforce the wrong normalization and reward recovery fails. |
| Feature rank | Linear AIRL2 requires full action-contrast rank. Raw design rank alone does not establish identification. |
| Feature condition number | Ill-conditioning makes reward directions poorly scaled and can destabilize or slow optimization. |
| State coverage | Reward recovery depends on the discriminator seeing transitions from most states. The three unobserved states in the primary cell are a known boundary. |
| State-action coverage | Rare action-state pairs are weakly identified. The minimum action share measures exposure to rare actions. |
| Segment behavioral separation | EM identifies segments only if the segments choose differently across states. Very similar segments require more data to separate. |
| Within-individual trajectory count | The consistency constraint requires at least two trajectories per individual to be useful; single-trajectory users contribute to the prior but not to within-user smoothing. |
| Transition row sums | Transition tensors must be row-stochastic in the $(A, S, S)$ orientation. |

## Primary Simulation Checks

Values from the primary `airl2_paper_identification` run recorded in
[airl2.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl2.json):

| Check | Value | Status |
| --- | --- | --- |
| Raw feature rank | 20 / 20 | context |
| Raw feature condition number | 20.411 | context |
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
correct integer indices for the specific environment. There are no defaults.
Omitting either argument raises a `TypeError`. An index outside the declared
state or action range raises a `ValueError`.

**Segment collapse.** One segment can absorb most of the prior mass if the
true behavioral difference is small or the initialization is unlucky. Monitor
`segment_priors` during EM; a prior near zero for any segment signals collapse.
The `prior_min`, `prior_damping`, and `prior_smoothing` settings resist collapse.

**EM stopping too early.** The primary cell meets the relative-change rule
after 2 iterations. Its log likelihood moves from -11609.662 to -11614.763
because the adversarial M-step is approximate. Inspect `em_log_likelihoods`
and the segment outputs rather than treating the convergence result alone as
evidence of a stable solution.
