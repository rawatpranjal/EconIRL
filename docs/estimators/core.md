# Core Estimators

EconIRL is a research build. It centers on one reference estimator, the nested
fixed point (NFXP), the exact maximum-likelihood estimator for tabular structural
dynamic discrete choice. NFXP is the one estimator with a verified paper-exact
replication, matched to Rust (1987) Table IX.

For the reasoning behind the split between this core and the other estimators, see
[Choosing an Estimator](landscape.md). For a side-by-side table, see
[Comparing Estimators](comparison.md).

| Estimator | Best for | Scope |
| --- | --- | --- |
| [NFXP](nfxp.md) | Exact tabular dynamic discrete choice, replicated to Rust (1987) Table IX. | Synthetic tabular simulation and the Rust bus replication. |

```{toctree}
:maxdepth: 1

landscape
comparison
nfxp
```
