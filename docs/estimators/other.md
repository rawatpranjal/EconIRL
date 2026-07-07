# Other Estimators

These estimators are available for advanced users, checks, and method development.
They sit outside the core focus, either because they overlap a core method, carry
caveated public status, or are narrower in scope. The grouping follows the sources
of complexity in [Choosing and Comparing Estimators](../comparing_estimators.md).

Read this page after the core menu. These methods are useful for specific
computational, behavioral, or diagnostic needs, but each carries a narrower
public evidence or interpretation boundary.

| Estimator | Source of complexity it answers | Use |
| --- | --- | --- |
| [NNES](nnes.md) | Large state space | Neural continuation value with finite reward parameters. |
| [MPEC](mpec.md) | Large state space | Constrained-optimization form of the DDC likelihood. |
| [UFXP](ufxp.md) | Large state space | Structural estimates without nesting a fixed point. |
| [RHIP](rhip.md) | Bounded or finite-horizon planning | Horizon-parameterised entropy IRL for route choice. |
| [f-IRL](f_irl.md) | Reward recovery via state-marginal matching | f-divergence state-marginal method. |
| [IQ-Learn](iq_learn.md) | Imitation and inverse soft-Q | Inverse soft-Q learning diagnostics. |

Research baselines under `econirl.contrib`: Max Margin Planning, GCL, GAIL,
Deep MaxEnt IRL, Bayesian IRL.

```{toctree}
:maxdepth: 1

nnes
mpec
ufxp
rhip
f_irl
iq_learn
```
