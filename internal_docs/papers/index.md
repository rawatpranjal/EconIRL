# Internal Paper Context

This directory replaces the old tracked PDF and TeX paper workspace for
agent-facing context. It should contain Markdown summaries, paper maps, and
estimator-specific reading notes. It should not contain PDFs, TeX sources, TeX
sidecars, manuscript drafts, or compiled outputs.

## Canonical Sources

| Area | Sources | Estimator folders |
| --- | --- | --- |
| Exact and CCP structural estimation | Rust (1987), Hotz and Miller (1993), Aguirregabiria and Mira (2002), Su and Judd (2012), Iskhakov et al. (2016) | `../estimators/nfxp/`, `../estimators/ccp/`, `../estimators/mpec/` |
| Approximate structural estimation | Luo and Sang (2024), Nguyen (2025), Adusumilli and Eckardt (2025) | `../estimators/sees/`, `../estimators/nnes/`, `../estimators/tdccp/` |
| Entropy IRL | Ng and Russell (2000), Abbeel and Ng (2004), Ziebart et al. (2008), Ziebart (2010), Wulfmeier et al. (2015) | `../estimators/maxent_irl/`, `../estimators/mce_irl/`, `../estimators/deep_mce_irl/`, `../estimators/deep_maxent_irl/`, `../estimators/max_margin_irl/` |
| Adversarial and Q-function IRL | Ho and Ermon (2016), Fu et al. (2018), Garg et al. (2021), Kang et al. (2025), Ni et al. (2020) | `../estimators/gail/`, `../estimators/airl/`, `../estimators/airl_het/`, `../estimators/iq_learn/`, `../estimators/gladius/`, `../estimators/f_irl/` |
| Identification and counterfactual foundations | Kim et al. (2021), Cao et al. (2021), Christensen and Connault style counterfactual sensitivity references | `../counterfactuals/design.md`, estimator folders with reward-gauge issues |

## Migration Rule

When a PDF, TeX primer, manuscript paragraph, or old paper note contains
durable estimator knowledge, move the idea into Markdown under the relevant
`internal_docs/estimators/<slug>/` folder. Do not restore the original binary
or TeX artifact.

Internal paper notes should answer:

- what object the paper identifies;
- what assumptions the estimator needs;
- what the package implements;
- what the package deliberately does not implement;
- what validation result or replication table supports the current claim;
- which public RTD page, if any, should receive the short user-facing version.

## Public Boundary

Public RTD pages cite papers through `docs/references.md` and only include the
amount of theory needed to use the estimator correctly. Full derivations,
replication notes, paper-vs-package differences, failed variants, and AI
maintenance context belong here or in the estimator folders.
