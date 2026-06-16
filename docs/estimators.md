# Estimators

Use this page to choose an estimator for your data and decision problem. The
evidence column states the current public scope.

For a side-by-side decision table, see [Comparing Estimators](estimators/comparison.md).

| Estimator | Best for | Scope |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | Exact tabular dynamic discrete choice. | Synthetic tabular simulation. |
| [CCP](estimators/ccp.md) | Hotz-Miller and NPL-style tabular DDC. | Synthetic tabular simulation with support conditions. |
| [MPEC](estimators/mpec.md) | Constrained-optimization check on the DDC likelihood. | Synthetic constrained-likelihood simulation. |
| [UFXP](estimators/ufxp.md) | Structural estimates at maximum-likelihood efficiency without nested solves. | Synthetic tabular simulation. |
| [NNES](estimators/nnes.md) | Neural value approximation inside NPL. | Synthetic low- and high-dimensional simulations. |
| [TD-CCP](estimators/tdccp.md) | Transition-density-free CCP parameter estimation with TD recursion. | Encoded-state finite-theta hard case with Algorithm 2 locally robust SEs. |
| [MCE-IRL](estimators/mce_irl.md) | Maximum causal entropy reward-feature matching. | Synthetic supplied-feature simulations. |
| [Deep MCE-IRL](estimators/deep_mce_irl.md) | Nonlinear reward-map recovery from known transitions. | Synthetic anchored neural reward-map simulations. |
| [AIRL](estimators/airl.md) | Adversarial state-reward recovery under original AIRL assumptions. | Synthetic state-only AIRL simulation. |
| [AIRL-Het](estimators/airl_het.md) | Anchored adversarial recovery with latent segments. | Synthetic serialized-content simulation. |
| [f-IRL](estimators/f_irl.md) | f-divergence state-marginal matching. | Synthetic state-marginal simulation. |
| [GLADIUS](estimators/gladius.md) | Neural Q and continuation modeling with anchor moments. | Preview: projected reward diagnostics. |
| [IQ-Learn](estimators/iq_learn.md) | Inverse soft-Q learning. | Preview: imitation and Q diagnostics. |

Each page states the target, evidence, and current scope. Preview pages are
for exploration, benchmarking, and method development.

## Other estimators (research code)

The package also ships implementations of several classic methods in
`econirl.contrib`. These are research code: they run on the uniform estimate
interface, but they are not benchmarked in the
[simulation studies](simulation_studies/index.md) and no validation evidence
is claimed for them.

| Estimator | Import | Tradition |
| --- | --- | --- |
| Max Margin Planning (MMP) | `econirl.contrib.max_margin_planning` | Ratliff et al structured margin. |
| GCL | `econirl.contrib.gcl` | Guided cost learning (Finn et al). |
| GAIL | `econirl.contrib.gail` | Adversarial imitation (Ho-Ermon). |
| Deep MaxEnt IRL | `econirl.contrib.deep_maxent_irl` | Neural reward feature matching (Wulfmeier et al). |
| Bayesian IRL | `econirl.contrib.bayesian_irl` | Posterior over rewards (Ramachandran-Amir). |

```{toctree}
:maxdepth: 1

estimators/comparison
estimators/nfxp
estimators/ccp
estimators/mpec
estimators/ufxp
estimators/nnes
estimators/tdccp
estimators/mce_irl
estimators/deep_mce_irl
estimators/airl
estimators/airl_het
estimators/f_irl
estimators/gladius
estimators/iq_learn
```
