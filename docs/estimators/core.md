# Core Estimators

EconIRL is a research build. These are the core estimators, the ones the project
focuses on. NFXP is the reference: the exact maximum-likelihood estimator for
tabular structural dynamic discrete choice, and the one with a verified paper-exact
replication, matched to Rust (1987) Table IX. The rest of the core spans the
structural and inverse-reinforcement-learning methods that carry the main
identification stories and method lineages.

For how to choose among them, including the side-by-side table, see
[Choosing an Estimator](landscape.md).

| Estimator | Family | Best for |
| --- | --- | --- |
| [NFXP](nfxp.md) | Structural | Exact tabular DDC, replicated to Rust (1987) Table IX. |
| [CCP](ccp.md) | Structural | Hotz-Miller and NPL tabular DDC without a nested solve. |
| [MCE-IRL](mce_irl.md) | IRL | Maximum causal entropy reward-feature matching. |
| [Neural MCE-IRL](deep_mce_irl.md) | IRL | Unrestricted neural reward map under the MCE objective. |
| [AIRL](airl.md) | IRL | Unified identified AIRL: state-only transfer, anchored action-dependent reward, and anchored latent heterogeneity. |
| [AIRL Anchored Heterogeneity](airl_het.md) | IRL | Detailed page for persistent latent segments under the anchored AIRL mode. |
| [GLADIUS](gladius.md) | IRL | Neural Q and continuation reward recovery at scale. |

GLADIUS is the core neural Q and continuation estimator.

## Identification

**What the data does not pin down.** Without explicit restrictions, the discount
factor, the shock distribution, and the reward level at the reference action are
not identified by observed choices alone (Magnac and Thesmar 2002, Proposition 2).
The table below gives the default status and the restriction that restores each.

| Object | Default status | What pins it |
| --- | --- | --- |
| Discount factor β | Not identified | Exclusion restriction or imposed parametrically |
| Shock distribution G | Not identified | Fixed by assumption (Gumbel is standard) |
| r(s, a_ref) | Not identified | Anchor normalization: r(s, a†) = 0 |
| r(s, a), a ≠ a_ref | Identified | Given the anchor and G |

In practice every estimator here fixes G to Type-I Gumbel and imposes an anchor
action. NFXP and CCP use the no-replacement action (a = 0). MCEIRL, AIRL, and
GenPQR apply the same convention.

**Equivalence to MaxEnt IRL.** Under Type-I Gumbel shocks and entropy coefficient
λ = 1, the DDC model and MaxEnt IRL are algebraically the same object (Kang 2026,
Theorem 2.7). The dictionary is exact: the Bellman operator T maps to the soft
Bellman operator T^soft (log-sum-exp); the conditional choice probability π(a|s)
maps to softmax(Q); the value function V(s) maps to log Σ_a exp Q(s, a). NFXP,
CCP, MCEIRL, AIRL, and GenPQR all solve the same fixed point. They differ in what
they treat as given and what they estimate from data.

```{toctree}
:maxdepth: 1

landscape
nfxp
ccp
mce_irl
deep_mce_irl
airl
airl_het
gladius
```
