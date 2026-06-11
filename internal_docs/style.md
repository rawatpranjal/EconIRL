# Documentation Style

## Internal Documentation

Internal documentation is for AIs and maintainers. It may be formal, dense, and
paper-level.

Estimator-level material belongs in `internal_docs/estimators/<slug>/`. Each
folder should keep deep implementation notes in `index.md`, paper and derivation
context in `papers.md`, and vertical source/docs/evidence links in `links.md`.

Internal estimator pages should include derivations, assumptions,
identification boundaries, objective functions, algorithms, implementation
contracts, validation design, result tables, known failure modes, and links to
machine-readable evidence.

Paper context should be Markdown. Do not restore PDFs, TeX files, compiled
documents, or paper build products to the package repository.

Internal docs should prefer precise claims over short claims. They should say
what is known, what is diagnostic only, and what is not supported.

## Public RTD Documentation

Public RTD documentation is for package users. It should be concise, linear,
and evidence-scoped.

Public pages should explain what the estimator does, when to use it, how to run
it, what evidence supports the page, and where to find the public API.

Keep vertical links from internal docs to public docs. Do not put internal
links or AI-only context in public source files, because RTD exposes `_sources`.

Public pages should not expose internal release-management terms, draft
planning language, paper build machinery, AI-only notes, or long derivations.

Avoid public phrases such as release claim, certified, validation target,
artifact, threshold check, and known truth. Prefer plain scientific language
such as simulation study, generated panel, held-back truth objects, result file,
and supported use.

## Shared Rules

Tables should contain real numbers from scripts or committed result files.

Do not repeat what a table already shows. Prose should interpret the result.

If evidence is diagnostic, call it diagnostic. Do not promote it to package
support.
