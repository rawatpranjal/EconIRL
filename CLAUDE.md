# EconIRL Agent Guide

EconIRL is a package-first repository. It exists to release and maintain the
`econirl` Python package that accompanies the ORE publication. It is not a
research workspace.

## Where To Look

Start with `internal_docs/index.md` before making non-trivial package, docs,
validation, data, or estimator changes.

Use `internal_docs/estimators/` for deep estimator context. Each estimator has
its own folder with `index.md`, `papers.md`, and `links.md`. These files are
the AI and maintainer source of truth for derivations, assumptions, validation
evidence, source-code links, public RTD state, and implementation contracts.

Use `internal_docs/papers/` for Markdown paper context. Do not restore PDFs,
TeX files, compiled documents, or manuscript workspaces to this package repo.

Use `docs/` only for public Read the Docs source. Public docs should be concise
and user-facing. They should not expose internal release notes, draft plans,
paper machinery, or AI-only context.

Use `validation/` for runnable validation scripts and machine-readable results.
Docs and tests should point to `validation/results/*.json`, not to paper
folders.

## Engineering Rules

Keep the public Python import surface stable unless the task explicitly asks
for an API migration.

Do not add neural reward support to structural estimators. Structural
estimators use linear utility by design. Neural reward plug-and-play applies to
IRL estimators.

Transition tensors must state their orientation. Estimator-facing code commonly
expects `(n_actions, n_states, n_states)`.

Before running an estimator on a new dataset, check feature rank, condition
number, state coverage, and single-action states. Stop if the feature matrix is
rank deficient.

Report validation honestly. Do not claim that all tests pass when any check
failed. Do not turn diagnostic evidence into release evidence.

## Documentation Rules

Internal docs may be dense, formal, and paper-level. They should preserve
derivations, design tradeoffs, result tables, and known failure modes.

Public RTD docs should be lean, linear, and evidence-scoped. Avoid internal
workflow language such as release claim, certified, validation target, artifact,
threshold check, or known truth.

For public RTD work, the live Read the Docs page is the authority after a push.
Local source checks alone are not enough for published documentation changes.

## Reproducibility Rules

Keep machine-readable validation evidence in `validation/results/`.

Keep runnable validation scripts in `validation/estimators/`.

Keep large raw data, local caches, generated builds, TeX products, PDFs, and
assistant state out of Git.

Use `python -m pytest tests/ -v -m "not slow"` for the fast suite. Use focused
tests when a migration only touches documentation, path contracts, or validation
metadata.
