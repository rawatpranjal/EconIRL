# Data Loading

Bundled data should be small enough to ship with the package and stable enough
to support examples, tests, or documented loaders.

Large raw datasets should not be tracked. They belong under ignored local data
directories or outside the repository.

Dataset loaders should make their source, schema, and cache behavior explicit.
They should fail with actionable messages when optional external data is
missing.

Test fixtures should live in `tests/fixtures/` when they are test-only. Package
sample data should live under `src/econirl/datasets/` only when it supports a
documented loader.
