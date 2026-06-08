# Estimator Map

Use this page to choose an estimator for your data and decision problem. The
status column describes the current validation scope for public examples and
external analysis.

## Structural Econometrics

| Estimator | Best for | Current status |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | Exact tabular dynamic discrete choice. | Validated reference estimator. |
| [CCP](estimators/ccp.md) | Hotz-Miller and NPL-style tabular DDC. | Validated with support conditions. |
| [MPEC](estimators/mpec.md) | Constrained-optimization check on the DDC likelihood. | Validated as an NFXP counterpart. |
| [SEES](estimators/sees.md) | Sieve value-function structural estimation. | Validated with optimizer-scope notes. |
| [NNES](estimators/nnes.md) | Neural value approximation inside NPL. | Validated on low- and high-dimensional cells. |
| [TD-CCP](estimators/tdccp.md) | Transition-free CCP estimation with TD recursion. | Validated for finite-parameter reward settings. |

## Inverse Reinforcement Learning

| Estimator | Best for | Current status |
| --- | --- | --- |
| [MCE-IRL](estimators/mce_irl.md) | Maximum causal entropy reward-feature matching. | Validated for supplied action-dependent reward features. |
| [Deep MCE-IRL](estimators/deep_mce_irl.md) | Nonlinear reward-map recovery from known transitions. | Validated for anchored neural reward maps. |
| [AIRL](estimators/airl.md) | Adversarial state-reward recovery under original AIRL assumptions. | Validated for state-only AIRL settings. |
| [AIRL-Het](estimators/airl_het.md) | Anchored adversarial recovery with latent segments. | Validated for the heterogeneous serialized-content setting. |
| [f-IRL](estimators/f_irl.md) | f-divergence state-marginal matching. | Validated for state-marginal reward recovery. |
| [GLADIUS](estimators/gladius.md) | Neural Q and continuation modeling with anchor moments. | Research preview for projected reward analysis. |
| [IQ-Learn](estimators/iq_learn.md) | Inverse soft-Q learning. | Research preview for imitation and Q diagnostics. |

## How to Read This Map

Validated pages state the target and the evidence behind it. Research preview
pages are useful for exploration, benchmarking, and method development, but
their pages also state the current scope of counterfactual evidence.

```{toctree}
:caption: Structural Econometrics
:maxdepth: 1

estimators/nfxp
estimators/ccp
estimators/mpec
estimators/sees
estimators/nnes
estimators/tdccp
```

```{toctree}
:caption: Inverse Reinforcement Learning
:maxdepth: 1

estimators/mce_irl
estimators/deep_mce_irl
estimators/airl
estimators/airl_het
estimators/f_irl
estimators/gladius
estimators/iq_learn
```
