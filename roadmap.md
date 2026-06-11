# EconIRL Package Migration Roadmap

Updated: 2026-06-11

This roadmap tracks the migration from a mixed package, papers, experiments,
and local research workspace into a package-first release repository. The
target repository contains the Python package, tests, public RTD source,
examples, validation evidence, and internal Markdown documentation for AIs and
maintainers.

The repository should not track PDFs, TeX manuscript sources, TeX compilation
products, literature dumps, local data, generated docs, local assistant state,
or draft research plans.

## Target Layout

```text
.
├── src/econirl/
├── tests/
├── docs/
├── internal_docs/
│   ├── index.md
│   ├── style.md
│   ├── api/
│   ├── estimators/
│   │   └── <estimator>/{index.md,papers.md,links.md}
│   ├── papers/
│   ├── data/
│   ├── counterfactuals/
│   └── validation/
├── validation/
│   ├── estimators/
│   └── results/
├── examples/
├── benchmarks/
├── scripts/
├── README.md
├── CLAUDE.md
├── pyproject.toml
└── roadmap.md
```

## Migration Principles

- Package source, tests, public docs, examples, and validation evidence are
  first-class.
- `papers/` is retired as an active project area.
- Deep estimator knowledge lives in
  `internal_docs/estimators/<estimator>/{index.md,papers.md,links.md}`.
- Paper context is retained as Markdown under `internal_docs/papers/` and
  estimator `papers.md` files, not as PDFs or TeX.
- Public RTD pages are separate from internal AI-maintainer documentation.
- Machine-readable validation evidence lives in `validation/results/*.json`.
- Runnable validation code lives in `validation/estimators/<estimator>/`.
- Generated PDFs, TeX files, compiler sidecars, docs builds, caches, local data,
  and assistant state are excluded from Git.

## Phase 1. Establish New Surfaces

Status: complete.

- Create `internal_docs/index.md` as the master map for estimator theory, API
  design, validation protocol, data loading, counterfactuals, and style rules.
- Create `internal_docs/style.md` with separate rules for internal documents and
  public RTD pages.
- Create one internal Markdown folder per estimator under
  `internal_docs/estimators/`, with `index.md`, `papers.md`, and `links.md`.
- Create API and design pages for package surface, data loading, estimator
  protocol, result objects, transitions, counterfactuals, and validation.
- Create `internal_docs/papers/index.md` as the Markdown replacement for the
  old tracked paper/PDF workspace.
- Rewrite root `CLAUDE.md` into a lean routing guide.
- Preserve the old long project context at
  `internal_docs/project_context_legacy.md` for audit and migration reference.

## Phase 2. Move Validation Evidence

Status: complete.

- Move estimator validation scripts from `papers/econirl_package/primers/` to
  `validation/estimators/`.
- Move machine-readable JSON results from paper folders to
  `validation/results/`.
- Move local smoke and support guards from `papers/econirl_package_jss/artifacts/`
  to `validation/estimators/` and `validation/results/`.
- Update validation scripts so their default JSON outputs target
  `validation/results/`.
- Stop tracking generated TeX result fragments.

## Phase 3. Update References

Status: complete.

- Update `docs/estimators/**/*.md` links from `papers/econirl_package/...` to
  `validation/...`.
- Update tests that read paper artifacts so they read validation results.
- Rename paper artifact tests to validation evidence tests when practical.
- Update experiment or benchmark references that still point at retired paper
  folders.
- Update README links only if they expose retired paths.

## Phase 4. Retire Paper Workspace

Status: complete.

- Remove `papers/` from tracked Git content after validation evidence and
  internal Markdown knowledge have been migrated.
- Do not keep PDFs, TeX manuscripts, compiled PDFs, `.aux`, `.log`, `.out`,
  `.bbl`, `.blg`, or similar paper build products.
- Convert retained estimator context into Markdown under `internal_docs/`,
  with estimator-specific paper context under each estimator folder.
- Keep publication manuscript material outside this package repository.

## Phase 5. Aggressive Cleanup

Status: complete.

- Expand `.gitignore` for Python caches, local tool state, environments,
  secrets, docs builds, temp files, backup files, data, PDFs, TeX products, and
  build artifacts.
- Keep retired root workspaces ignored with root-anchored patterns such as
  `/papers/` and `/experiments/`, so `internal_docs/papers/` remains trackable.
- Remove tracked `.DS_Store`, Python bytecode, root logs, docs build output,
  LaTeX sidecars, compiled PDFs, and local artifacts.
- Keep only small bundled package data needed by loaders, tests, or examples.
- Keep large raw data ignored and outside Git.

## Phase 6. Verification

Status: complete for migration gates.

Run a path audit.

```bash
git ls-files | rg '(^papers/|^experiments/|^docs/archive/|^docs/_build/|^data/raw/|\.aux$|\.bbl$|\.blg$|\.log$|\.out$|\.fdb_latexmk$|\.fls$|\.DS_Store$|__pycache__|texput\.log$|\.pdf$|\.tex$)'
```

Expected result: no output.

Run focused migration tests.

```bash
python -m pytest tests/test_package_imports.py tests/test_rtd_style_guide.py tests/test_validation_evidence.py tests/test_nfxp_release_artifact.py tests/test_mpec.py tests/integration/test_gcl_estimation.py::TestGCLSimpleMDP::test_gcl_simple_3state_mdp tests/integration/test_gcl_estimation.py::TestGCLSimpleMDP::test_gcl_learns_replacement_pattern -q
```

The focused migration suite passed locally on 2026-06-11 with 42 tests passing
and 5 warnings.

Build public docs locally.

```bash
LC_ALL=C LANG=C python -m sphinx -b html docs docs/_build/html
```

The local Sphinx build passed on 2026-06-11.

Verify the internal estimator folder contract.

```bash
for d in internal_docs/estimators/*/; do
  for f in index.md papers.md links.md; do
    test -f "$d$f" || echo "missing $d$f"
  done
done
```

Expected result: no output. This check passed locally on 2026-06-11.

Verify internal relative links and `.gitignore` behavior.

```bash
git check-ignore -v internal_docs/papers/index.md internal_docs/estimators/nfxp/papers.md
```

Expected result: no output. This check passed locally on 2026-06-11.

The internal Markdown link audit also passed locally on 2026-06-11.

Run the fast suite when path migration is stable.

```bash
python -m pytest tests/ -v -m "not slow"
```

The broad fast suite was started during migration verification, exposed the
missing `econirl.estimation.gcl` compatibility module, and was stopped after
the actionable failure was isolated. The focused GCL compatibility tests now
pass.

For public RTD changes that are pushed, manually trigger Read the Docs and
verify the live cache-busted pages.

## Remaining Follow-Up

- Decide which example directories remain public examples and which become
  internal validation fixtures.
- Add release artifacts before promoting contrib/internal estimators to public
  RTD. All estimator folders now have substantive internal notes. The public
  estimator set has target-depth notes; MaxEnt IRL has a hidden diagnostic
  page; GAIL, GCL, Bayesian IRL, Max Margin IRL, Max Margin Planning,
  Behavioral Cloning, and Deep MaxEnt IRL have internal pages that explicitly
  record their public-doc boundary and validation gaps.
- Trigger and verify public RTD only after these source changes are committed
  and pushed.
