# Estimators

Use this page to choose an estimator for your data and decision problem. The
core list focuses on the most useful, identified or explicitly normalized
targets, while still covering the main application domains.

For a side-by-side decision table, see [Comparing Estimators](estimators/comparison.md).

## Core estimators

| Estimator | Best for | Scope |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | Exact tabular dynamic discrete choice. | Synthetic tabular simulation. |
| [CCP](estimators/ccp.md) | Hotz-Miller and NPL-style tabular DDC. | Synthetic tabular simulation with support conditions. |
| [NNES](estimators/nnes.md) | Neural value approximation inside NPL. | Synthetic low- and high-dimensional simulations. |
| [TD-CCP](estimators/tdccp.md) | Transition-density-free CCP parameter estimation with TD recursion. | Encoded-state finite-theta hard case with Algorithm 2 locally robust SEs. |
| [MCE-IRL](estimators/mce_irl.md) | Maximum causal entropy reward-feature matching. | Synthetic supplied-feature simulations. |
| [RHIP](estimators/rhip.md) | Horizon-parameterised entropy IRL spanning MCE-IRL to Max-Margin Planning. | Synthetic route-choice horizon spectrum. |
| [AIRL](estimators/airl.md) | AIRL-1: adversarial state-reward recovery under original AIRL assumptions. | Synthetic state-only AIRL simulation. |
| [AIRL-Het](estimators/airl_het.md) | AIRL-2: anchored adversarial recovery with latent segments. | Synthetic serialized-content simulation. |
| [GLADIUS](estimators/gladius.md) | Neural Q and continuation modeling with anchor moments. | Preview: projected reward diagnostics. |

Each core page states the target, evidence, and current scope.

## Other estimators

These estimators remain available for advanced users, checks, and method
development, but they are not the default paper-replication or release-focus
surface.

| Estimator | Use | Current role |
| --- | --- | --- |
| [MPEC](estimators/mpec.md) | Constrained-optimization check on the DDC likelihood. | Secondary structural check; overlaps with NFXP/CCP and has higher solver complexity. |
| [UFXP](estimators/ufxp.md) | Structural estimates at maximum-likelihood efficiency without nested solves. | Secondary structural speed/FOC variant. |
| [Deep MCE-IRL](estimators/deep_mce_irl.md) | Nonlinear reward-map recovery from known transitions. | Neural extension of the core MCE-IRL path. |
| [f-IRL](estimators/f_irl.md) | f-divergence state-marginal matching. | Narrower state-marginal method. |
| [IQ-Learn](estimators/iq_learn.md) | Inverse soft-Q learning. | Preview: imitation and Q diagnostics. |
| Max Margin Planning (MMP) | `econirl.contrib.max_margin_planning` | Research baseline. |
| GCL | `econirl.contrib.gcl` | Research baseline. |
| GAIL | `econirl.contrib.gail` | Research baseline. |
| Deep MaxEnt IRL | `econirl.contrib.deep_maxent_irl` | Research baseline. |
| Bayesian IRL | `econirl.contrib.bayesian_irl` | Research baseline. |

```{toctree}
:maxdepth: 1

estimators/comparison
estimators/nfxp
estimators/ccp
estimators/nnes
estimators/tdccp
estimators/mce_irl
estimators/rhip
estimators/airl
estimators/airl_het
estimators/gladius
estimators/mpec
estimators/ufxp
estimators/deep_mce_irl
estimators/f_irl
estimators/iq_learn
```
