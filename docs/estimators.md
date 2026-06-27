# Estimators

EconIRL is a research build. It centers on one reference estimator, the nested
fixed point (NFXP), and treats the rest as answers to specific complications that
break NFXP's canonical case. For the reasoning behind this split, see
[Choosing an Estimator](estimators/landscape.md). For a side-by-side decision
table, see [Comparing Estimators](estimators/comparison.md).

## Core estimator

| Estimator | Best for | Scope |
| --- | --- | --- |
| [NFXP](estimators/nfxp.md) | Exact tabular dynamic discrete choice, replicated to Rust (1987) Table IX. | Synthetic tabular simulation and the Rust bus replication. |

## Other estimators

Each of these relaxes one assumption that makes NFXP exact and cheap. They remain
available for advanced users, checks, and method development. The grouping follows
the sources of complexity in [Choosing an Estimator](estimators/landscape.md).

| Estimator | Source of complexity it answers | Use |
| --- | --- | --- |
| [CCP](estimators/ccp.md) | Large state space | Hotz-Miller and NPL tabular DDC without a nested solve. |
| [MPEC](estimators/mpec.md) | Large state space | Constrained-optimization form of the DDC likelihood. |
| [UFXP](estimators/ufxp.md) | Large state space | Structural estimates without nesting a fixed point. |
| [NNES](estimators/nnes.md) | Large state space | Neural continuation value with finite reward parameters. |
| [TD-CCP](estimators/tdccp.md) | Hard-to-model transition density | Reward parameters without modeling the transition density. |
| [MCE-IRL](estimators/mce_irl.md) | Unknown reward form | Maximum causal entropy reward-feature matching. |
| [Neural MCE-IRL](estimators/deep_mce_irl.md) | Unknown reward form | Unrestricted neural reward map under the MCE objective. |
| [AIRL](estimators/airl.md) | Unknown reward form | Adversarial transferable state-only reward. |
| [GLADIUS](estimators/gladius.md) | Unknown reward form | Neural Q and continuation reward recovery at scale. |
| [AIRL-Het](estimators/airl_het.md) | Latent heterogeneity | Segment-specific rewards under an anchor. |
| [RHIP](estimators/rhip.md) | Bounded or finite-horizon planning | Horizon-parameterised entropy IRL for route choice. |
| [f-IRL](estimators/f_irl.md) | Reward recovery via state-marginal matching | f-divergence state-marginal method. |
| [IQ-Learn](estimators/iq_learn.md) | Imitation and inverse soft-Q | Inverse soft-Q learning diagnostics. |

Research baselines under `econirl.contrib`: Max Margin Planning, GCL, GAIL,
Deep MaxEnt IRL, Bayesian IRL.

```{toctree}
:maxdepth: 1

estimators/landscape
estimators/comparison
```

```{toctree}
:maxdepth: 1
:caption: Core

estimators/nfxp
```

```{toctree}
:maxdepth: 1
:caption: Other

estimators/ccp
estimators/mpec
estimators/ufxp
estimators/nnes
estimators/tdccp
estimators/mce_irl
estimators/deep_mce_irl
estimators/airl
estimators/gladius
estimators/airl_het
estimators/rhip
estimators/f_irl
estimators/iq_learn
```
