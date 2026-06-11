# Internal Estimator Catalog

This catalog is the maintainer and AI map for estimator-specific knowledge. It
is intentionally broader than public Read the Docs. Some estimators are public
RTD pages, some are package APIs without a public guide, and some are contrib
or compatibility surfaces retained for users and tests.

Each estimator folder follows this layout:

- `index.md` for deep implementation and validation context;
- `papers.md` for source papers, old primer context, and derivation tasks;
- `links.md` for vertical links to package modules, public RTD source, live RTD
  page when shown, validation runners, and result JSON.

Public RTD pages should not copy these internal pages. They should draw only
the concise user-facing subset.

## Public RTD Estimators

| Estimator | Internal context | Paper context | Public source | Validation |
| --- | --- | --- | --- | --- |
| NFXP | `nfxp/index.md` | `nfxp/papers.md` | `../../docs/estimators/nfxp.md` | `../../validation/results/nfxp.json` |
| CCP | `ccp/index.md` | `ccp/papers.md` | `../../docs/estimators/ccp.md` | `../../validation/results/ccp.json` |
| MPEC | `mpec/index.md` | `mpec/papers.md` | `../../docs/estimators/mpec.md` | `../../validation/results/mpec.json` |
| NNES | `nnes/index.md` | `nnes/papers.md` | `../../docs/estimators/nnes.md` | `../../validation/results/nnes.json` |
| TD-CCP | `tdccp/index.md` | `tdccp/papers.md` | `../../docs/estimators/tdccp.md` | `../../validation/results/tdccp.json` |
| MCE-IRL | `mce_irl/index.md` | `mce_irl/papers.md` | `../../docs/estimators/mce_irl.md` | `../../validation/results/mce_irl.json` |
| Deep MCE-IRL | `deep_mce_irl/index.md` | `deep_mce_irl/papers.md` | `../../docs/estimators/deep_mce_irl.md` | `../../validation/results/deep_mce_irl.json` |
| AIRL | `airl/index.md` | `airl/papers.md` | `../../docs/estimators/airl.md` | `../../validation/results/airl.json` |
| AIRL-Het | `airl_het/index.md` | `airl_het/papers.md` | `../../docs/estimators/airl_het.md` | `../../validation/results/aairl.json` |
| f-IRL | `f_irl/index.md` | `f_irl/papers.md` | `../../docs/estimators/f_irl.md` | `../../validation/results/f_irl.json` |
| GLADIUS | `gladius/index.md` | `gladius/papers.md` | `../../docs/estimators/gladius.md` | `../../validation/results/gladius.json` |
| IQ-Learn | `iq_learn/index.md` | `iq_learn/papers.md` | `../../docs/estimators/iq_learn.md` | `../../validation/results/iq_learn.json` |

## Hidden Or Internal-Only Estimators

| Estimator | Internal context | Paper context | Public state | Validation |
| --- | --- | --- | --- | --- |
| SEES | `sees/index.md` | `sees/papers.md` | Source exists at `../../docs/estimators/sees.md`, but Sphinx currently excludes it. | `../../validation/results/sees.json` |
| MaxEnt IRL | `maxent_irl/index.md` | `maxent_irl/papers.md` | No current public RTD estimator page. | `../../validation/results/maxent_irl.json` |
| Deep MaxEnt IRL | `deep_maxent_irl/index.md` | `deep_maxent_irl/papers.md` | No current public RTD estimator page. | Contrib implementation only. |
| Max Margin IRL | `max_margin_irl/index.md` | `max_margin_irl/papers.md` | Top-level package API, no current public RTD estimator page. | No current validation JSON. |
| Max Margin Planning | `max_margin_planning/index.md` | `max_margin_planning/papers.md` | Contrib implementation only. | No current validation JSON. |
| GAIL | `gail/index.md` | `gail/papers.md` | Contrib implementation only. | `../../validation/estimators/gail/run.py` |
| GCL | `gcl/index.md` | `gcl/papers.md` | Sklearn wrapper and contrib implementation, no current public RTD estimator page. | `../../validation/results/gcl.json` |
| Bayesian IRL | `bayesian_irl/index.md` | `bayesian_irl/papers.md` | Contrib implementation only. | No current validation JSON. |
| Behavioral Cloning | `behavioral_cloning/index.md` | `behavioral_cloning/papers.md` | Lower-level baseline API, no current public RTD estimator page. | Unit tests only. |

## Depth Status

| Status | Estimators | Meaning |
| --- | --- | --- |
| Release-depth internal page | NFXP, CCP, MPEC, NNES, TD-CCP, SEES, MCE-IRL, Deep MCE-IRL, AIRL, AIRL-Het, GLADIUS, IQ-Learn, f-IRL | `index.md` includes derivations, estimator mechanics, identification checks, validation design, result tables, counterfactual interpretation, debugging order, caveats, and implementation links. Some pages explicitly record partial or failed structural certification. |
| Hidden diagnostic internal page | MaxEnt IRL | Internal page is deep enough for maintainers, but the estimator has no current public RTD page and is framed as a diagnostic baseline. |
| Contrib/internal page with release gap | GAIL, GCL, Bayesian IRL, Max Margin IRL, Max Margin Planning, Behavioral Cloning, Deep MaxEnt IRL | Internal page explains the algorithm, implementation paths, tests or artifacts, public-doc boundary, and missing validation needed before public exposure. |

## Completion Standard

An estimator folder is complete only when its `index.md`, `papers.md`, and
`links.md` jointly explain the estimator target, implementation modules,
paper assumptions, validation evidence or validation gap, public documentation
state, and known failure modes. If an estimator has no release artifact, the
page must say that directly instead of implying public readiness.
