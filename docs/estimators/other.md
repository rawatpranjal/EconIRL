# Contrib and Research Estimators

These estimators sit outside the core roster. They are importable from
`econirl.contrib` for advanced users, cross-checks, and method development. Each
one either overlaps a core method, is narrower in scope, or carries a more limited
public status. For how they relate to the core methods, see
[Choosing an Estimator](landscape.md).

| Estimator | Source of complexity it answers | Use |
| --- | --- | --- |
| [TD-CCP](tdccp.md) | Large state space | Reward parameters without modeling the transition density. |
| [NNES](nnes.md) | Large state space | Neural continuation value with finite reward parameters. |
| [MPEC](mpec.md) | Large state space | Constrained-optimization form of the DDC likelihood. |
| [UFXP](ufxp.md) | Large state space | Structural estimates without nesting a fixed point. |
| [RHIP](rhip.md) | Bounded or finite-horizon planning | Horizon-parameterised entropy IRL for route choice. |
| [f-IRL](f_irl.md) | Reward recovery via state-marginal matching | f-divergence state-marginal method. |
| [IQ-Learn](iq_learn.md) | Imitation and inverse soft-Q | Inverse soft-Q learning diagnostics. |

Research baselines also under `econirl.contrib`: Max Margin Planning, GCL, GAIL,
Deep MaxEnt IRL, Bayesian IRL.

```{toctree}
:maxdepth: 1

tdccp
nnes
mpec
ufxp
rhip
f_irl
iq_learn
```
