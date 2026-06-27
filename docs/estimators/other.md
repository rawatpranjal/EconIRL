# Other Estimators

Each of these relaxes one assumption that makes NFXP exact and cheap. They remain
available for advanced users, checks, and method development. The grouping follows
the sources of complexity in [Choosing an Estimator](landscape.md).

| Estimator | Source of complexity it answers | Use |
| --- | --- | --- |
| [CCP](ccp.md) | Large state space | Hotz-Miller and NPL tabular DDC without a nested solve. |
| [MPEC](mpec.md) | Large state space | Constrained-optimization form of the DDC likelihood. |
| [UFXP](ufxp.md) | Large state space | Structural estimates without nesting a fixed point. |
| [NNES](nnes.md) | Large state space | Neural continuation value with finite reward parameters. |
| [TD-CCP](tdccp.md) | Hard-to-model transition density | Reward parameters without modeling the transition density. |
| [MCE-IRL](mce_irl.md) | Unknown reward form | Maximum causal entropy reward-feature matching. |
| [Neural MCE-IRL](deep_mce_irl.md) | Unknown reward form | Unrestricted neural reward map under the MCE objective. |
| [AIRL](airl.md) | Unknown reward form | Adversarial transferable state-only reward. |
| [GLADIUS](gladius.md) | Unknown reward form | Neural Q and continuation reward recovery at scale. |
| [AIRL-Het](airl_het.md) | Latent heterogeneity | Segment-specific rewards under an anchor. |
| [RHIP](rhip.md) | Bounded or finite-horizon planning | Horizon-parameterised entropy IRL for route choice. |
| [f-IRL](f_irl.md) | Reward recovery via state-marginal matching | f-divergence state-marginal method. |
| [IQ-Learn](iq_learn.md) | Imitation and inverse soft-Q | Inverse soft-Q learning diagnostics. |

Research baselines under `econirl.contrib`: Max Margin Planning, GCL, GAIL,
Deep MaxEnt IRL, Bayesian IRL.

```{toctree}
:maxdepth: 1

ccp
mpec
ufxp
nnes
tdccp
mce_irl
deep_mce_irl
airl
gladius
airl_het
rhip
f_irl
iq_learn
```
