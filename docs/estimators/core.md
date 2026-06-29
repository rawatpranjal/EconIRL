# Core Estimators

EconIRL is a research build. These are the core estimators, the ones the project
focuses on. NFXP is the reference: the exact maximum-likelihood estimator for
tabular structural dynamic discrete choice, and the one with a verified paper-exact
replication, matched to Rust (1987) Table IX. The rest of the core spans the
structural and inverse-reinforcement-learning methods that carry the main
identification stories and method lineages.

For how to choose among them, including the side-by-side table, see
[Choosing and Comparing Estimators](../comparing_estimators.md).

| Estimator | Family | Best for |
| --- | --- | --- |
| [NFXP](nfxp.md) | Structural | Exact tabular DDC, replicated to Rust (1987) Table IX. |
| [CCP](ccp.md) | Structural | Hotz-Miller and NPL tabular DDC without a nested solve. |
| [TD-CCP](tdccp.md) | Structural | Reward parameters without modeling the transition density. |
| [MCE-IRL](mce_irl.md) | IRL | Maximum causal entropy reward-feature matching. |
| [Neural MCE-IRL](deep_mce_irl.md) | IRL | Unrestricted neural reward map under the MCE objective. |
| [AIRL](airl.md) | IRL | Adversarial state-only reward recovery under the original AIRL transfer assumptions. |
| [AIRL-Het](airl_het.md) | IRL | Segment-specific action-dependent rewards under exit and absorbing-state anchors. |
| [GLADIUS](gladius.md) | IRL | Neural Q and continuation reward recovery at scale. |

GLADIUS is the core neural Q and continuation estimator.

```{toctree}
:maxdepth: 1

nfxp
ccp
tdccp
mce_irl
deep_mce_irl
airl
airl_het
gladius
```
