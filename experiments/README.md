# experiments/ — local research one-offs

One-off experiment and sweep drivers: the scripts you run to investigate a
question, not the maintained tooling that builds the public docs. The home and
this README are tracked so the convention is visible; the **contents are
gitignored** (`/experiments/*`, `!/experiments/README.md`).

One level under the repo root on purpose: these scripts use relative paths like
`../src`, `../docs`, `../examples`, so they keep working here exactly as they did
in `scripts/`.

## What belongs here (guidance, not law)

- hyperparameter sweeps, sample-size sweeps, ablations.
- one-off comparison experiments and their drivers (autolab/SML/NGSIM runners).
- anything exploratory you would not ship in the package.

## What does NOT belong here

- `scripts/sim_*.py`, `quick_all_estimators.py` — those **generate public docs
  pages**, are maintained, and stay in `scripts/`.
- `scripts/download_*` — data fetchers, also maintained tooling.
- generated results — those go to `outputs/`. Code here, artifacts there.
