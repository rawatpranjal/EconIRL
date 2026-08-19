# Core Estimators

## Important Links

- [Choosing and Comparing Estimators](../comparing_estimators.md)
- [Other Estimators](other.md)
- [Simulation Studies](../simulation_studies/index.md)
- [Paper Replications](../replications.md)

Core contains the project's main methods. NFXP is the exact maximum-likelihood
reference for tabular structural dynamic discrete choice, with a verified
paper-exact match to Rust (1987) Table IX. The remaining methods carry the main
structural and inverse-reinforcement-learning identification lines.

| Estimator | Family | Best for |
| --- | --- | --- |
| [NFXP](nfxp.md) | Structural | Exact tabular DDC, the maximum-likelihood reference. |
| [CCP](ccp.md) | Structural | Hotz-Miller and NPL tabular DDC without a nested solve. |
| [MCE-IRL](mce_irl.md) | IRL | Maximum causal entropy reward-feature matching. |
| [Neural MCE-IRL](deep_mce_irl.md) | IRL | Unrestricted neural reward map under the MCE objective. |
| [AIRL](airl.md) | IRL | Adversarial state-only reward recovery under the original AIRL transfer assumptions. |
| [NeuralAIRL](neural_airl.md) | IRL | Nonlinear state reward recovery with neural reward, shaping, and policy functions. |
| [GLADIUS](gladius.md) | IRL | Neural Q and continuation reward recovery at scale. |

```{toctree}
:maxdepth: 1

nfxp
ccp
mce_irl
deep_mce_irl
airl
neural_airl
gladius
```
