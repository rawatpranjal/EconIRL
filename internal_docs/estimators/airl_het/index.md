# AIRL-Het Internal Notes

AIRL-Het is the anchored heterogeneous extension of AIRL used for
serialized-content style dynamic choice settings. It adds latent segments,
segment-specific reward and policy objects, an EM loop, an exit-action reward
anchor, and an absorbing-state value normalization. In this package it is the
adversarial estimator that is allowed to make action-dependent structural claims
under the current validation evidence.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- AIRL base context: `../airl/index.md`.
- Public RTD source: `../../../docs/estimators/airl_het.md`.
- Lower-level estimator: `../../../src/econirl/estimation/adversarial/airl_het.py`.
- Validation runner: `../../../validation/estimators/aairl/run.py`.
- Validation result: `../../../validation/results/aairl.json`.

## Model

Each individual belongs to one latent segment `k`. Segment `k` has its own
reward, shaping potential, policy, and prior probability:

```text
f_k(s, a, s') = g_k(s, a) + beta h_k(s') - h_k(s)
```

with discriminator

```text
D_k(s, a, s') = exp(f_k(s, a, s'))
                / (exp(f_k(s, a, s')) + pi_k(a | s)).
```

The mixture layer tracks segment priors and trajectory-level posterior weights:

```text
lambda_k = prior segment probability
q_ik     = Pr(segment k | trajectory i).
```

The EM loop alternates between assigning trajectories to segments and updating
segment-specific adversarial rewards and policies.

## Anchors

The action-dependent AIRL identification problem is handled through two
normalizations:

```text
g_k(s, exit) = 0 for every state s and segment k
h_k(absorbing_state) = 0 for every segment k.
```

The exit-action anchor pins reward levels. The absorbing-state anchor pins the
value potential. Together they remove the residual degrees of freedom that make
standard AIRL fail in action-dependent DDC environments with absorbing exits.

## Validation Status

The current artifact status is `pass` on the anchored serialized-content latent
segment cell `airl_het_paper_identification`.

| Check | Current value | Gate |
| --- | ---: | --- |
| Converged | true | pass |
| EM iterations | 2 | pass |
| Observations | 12368 | context |
| Segments | 2 | pass |
| Segment assignment accuracy | 0.895 | pass |
| Segment prior L1 | 0.0435 | pass |
| Max segment reward NRMSE | 0.265 | pass |
| Max segment policy TV | 0.0591 | pass |
| Max segment value NRMSE | 0.142 | pass |
| Max segment Q NRMSE | 0.211 | pass |
| Type A max regret | 0.0145 | pass |
| Type B max regret | 0.1189 | pass |
| Type C max regret | 0.00687 | pass |

Support diagnostics:

- observed states: 58/61;
- state-action coverage: 0.934;
- feature rank: 20/20;
- condition number: 20.411;
- minimum action share: 0.204;
- maximum transition row error: 0.0;
- anchor valid: true.

The warning `58 of 61 states are observed` is expected validation context, not a
gate failure in this artifact. It should remain visible in internal and public
docs because heterogeneity claims are support-sensitive.

## Interpretation For Maintainers

AIRL-Het has a stronger structural claim than base AIRL only because the anchor
design and heterogeneity structure are part of the estimator contract. The
maintained interpretation is:

- segment priors and assignment accuracy are validation objects;
- reward, policy, value, Q, and counterfactual metrics are segment-level
  objects;
- the estimator is tied to panel settings where trajectory-level segment
  assignment is meaningful;
- the package context may reference serialized-content DDC, but manuscript
  drafting and publication-specific assets stay outside this repo.

## Debugging Order

1. Verify exit action and absorbing state indices.
2. Verify segment count, initialization, and prior smoothing.
3. Confirm trajectory-to-individual grouping before checking segment accuracy.
4. Check EM log-likelihood and posterior assignments.
5. Inspect segment-specific reward anchors.
6. Compare segment reward maps, policies, values, and Q functions under the
   estimated-to-true segment permutation.
7. Review counterfactual regret by segment and perturbation type.

## Implementation Paths

- Lower-level estimator: `../../../src/econirl/estimation/adversarial/airl_het.py`.
- Validation runner: `../../../validation/estimators/aairl/run.py`.
- Validation JSON: `../../../validation/results/aairl.json`.
- Public docs: `../../../docs/estimators/airl_het.md`.

## Public Documentation Boundary

Public RTD should describe the estimator contract, required panel structure,
anchors, segment outputs, and validation receipt. It should not include
manuscript production notes, paper assets, or broader ORE drafting context.
