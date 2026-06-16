# Status: econirl - 2026-06-15

## This month (north-star)
EconIRL ships a lean, public Python package plus a Read the Docs site that accompanies a journal publication. Locally it is a research mono-repo (papers, experiments, retired code stay private and out of public history). This month's work is a synthetic "forms" framework plus a two-tier set of simulation studies, so the package is trustworthy to machine-learning engineers and scientists on known-truth, all-synthetic ground. Alongside it: a small batch of public-docs fixes and an optimizer-library migration.

## Active chunks
All feature streams are designed and verified; none are built yet. Package migration is fully done.

- **D1/D2/D3 doc fixes** - scoped + explored, build queued first. Public style-guide under-enforces (95+ live violations), the index page contradicts per-page framing, and a solver bug shows on one page. Mechanical, no estimator risk.
- **F0-F1 forms registry + loader** - design done, build pending. Pure data, no estimator risk - the natural first feature.
- **F2 graph generator (road_network)** - design done, build pending.
- **F3 sim-studies reorg** - design done; replaces abstract toy problems with real dynamic-choice case studies (Bus Engine, Gridworld, Fleet Maintenance, and more).
- **F4 fit() unlock** - design done, BLOCKED on a sign-off (touches the public interface).
- **F5 public forms API** - design ~70%, blocked behind F4 and the same sign-off.
- **R5 jaxopt migration** - design done, build pending (swap the optimizer library with agreement tests).

## Needs your attention (ranked)
1. **F4/F5 public interface names + signatures** need your approval before anything is exposed. This is the one human gate blocking the riskier half of the feature work.
2. **Last simulation-suite audit passed *with findings*** (4 days ago) - three honesty gaps worth a look: one estimator's standard-error failures (70% of runs) shown as a bland "n/a", a cell that claims scaling stress the data does not show, and an unexplained non-convergence. Worth fixing before these pages go further.
3. Nothing else is blocked - the doc fixes and forms work can start immediately.

## Shipped in last 7 days
- Wired the autonomous build engine (hooks loaded, planning files stay private).
- Reorged the repo into a clean published-lean / research-local layout (removed the redundant outputs folder; experiments now sit beside their results).
- Earlier in the window: public docs pages for several estimators, neural variants, and four simulation studies, plus style-gate fixes.

## Recent commits (7 days)
```
c74657d9 known_truth.py default output-dir -> validation/artifacts (outputs/ removed)
9a8d1907 Drop outputs/ (redundant with RTD); experiment results sit beside their driver in experiments/
7134d1bf Move 6 experiment/sweep drivers scripts/ -> experiments/ (local); scripts/ now maintained tooling only
7e9d9415 neural_mpec_experiment writes to outputs/ not scripts/
8d400bb3 Consolidate: drop top-level benchmarks/ (dead torch-vs-jax) -> docs/research; move stray result dumps
```

## Suggested next move
Run the build engine - it starts on the public-docs fixes (zero estimator risk), then the forms registry, stopping cleanly at the interface sign-off gate.

## TLDR
The project is at a clean, fully-planned starting line. Everything for this month's work - a synthetic testing framework and a fresh set of realistic simulation studies - is designed, reviewed, and signed off, but no feature code has been written yet. The package itself is finished and stable. Since the last check-in, the team mainly set up the automated build process and tidied the repository so private research stays private and the public release stays lean. The one thing waiting on a decision from you is approval of the names and shape of a new public interface, which gates the riskier half of the work; the rest can start right away. The recommended next step is to kick off the build, which will safely knock out the public-documentation fixes first, then start the framework, and pause when it reaches the part that needs your sign-off. One quality note: the most recent review of the simulation studies passed but flagged three spots where results were glossed over rather than explained, worth cleaning up soon.
