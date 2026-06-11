# EconIRL Internal Documentation Index

This directory is the AI and maintainer knowledge base for the package. It is
not the public documentation surface.

## Core Maps

- `style.md` defines internal and external documentation rules.
- `papers/index.md` defines how paper context is retained as Markdown without
  tracking PDFs or TeX.
- `api/package_surface.md` defines stable public imports and advanced modules.
- `api/data_loading.md` defines bundled data, external data, and cache policy.
- `api/estimator_protocol.md` defines estimator interface expectations.
- `api/results_objects.md` defines summary and result object expectations.
- `api/transitions.md` defines transition tensor conventions.
- `counterfactuals/design.md` defines the cross-estimator counterfactual model.
- `validation/validation_protocol.md` defines validation script and JSON rules.
- `estimators/index.md` maps every estimator folder to source code, validation
  evidence, paper context, and public RTD state.

## Estimator Context

- `estimators/nfxp/`
- `estimators/ccp/`
- `estimators/mpec/`
- `estimators/nnes/`
- `estimators/tdccp/`
- `estimators/sees/`
- `estimators/mce_irl/`
- `estimators/deep_mce_irl/`
- `estimators/maxent_irl/`
- `estimators/deep_maxent_irl/`
- `estimators/max_margin_irl/`
- `estimators/max_margin_planning/`
- `estimators/airl/`
- `estimators/airl_het/`
- `estimators/gail/`
- `estimators/gcl/`
- `estimators/gladius/`
- `estimators/iq_learn/`
- `estimators/f_irl/`
- `estimators/bayesian_irl/`
- `estimators/behavioral_cloning/`

Each estimator folder should contain `index.md`, `papers.md`, and `links.md`.
The `links.md` file is the direct vertical link between internal context,
package source, validation evidence, and public RTD source or live pages. Public
RTD pages may draw from these files, but should not copy their full density or
internal maintenance notes.

## Legacy Context

`project_context_legacy.md` preserves the pre-migration root `CLAUDE.md`. Use
it only as historical context while moving durable material into the focused
internal docs above.

`validation/estimator_fix_blueprint_legacy.md` preserves the old estimator
release blueprint. Treat it as historical guidance only; active validation
paths now live under `validation/`.
