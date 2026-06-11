# Package Surface

The stable public package surface is the top-level `econirl` import namespace
and the sklearn-style estimator API.

Expected public imports include `NFXP`, `CCP`, `MPEC`, `NNES`, `TDCCP`,
`MCEIRL`, `MaxEntIRL`, `MaxMarginIRL`, `GLADIUS`, `NeuralGLADIUS`, `AIRL`,
`NeuralAIRL`, `IQLearn`, `MCEIRLNeural`, `TransitionEstimator`, `LinearCost`,
and `make_utility`.

The complete estimator routing table lives in `../estimators/index.md`. Use it
before changing estimator imports, public RTD pages, validation evidence, or
contrib compatibility shims.

Lower-level modules under `econirl.estimation`, `econirl.core`,
`econirl.environments`, `econirl.preferences`, and `econirl.inference` remain
available for advanced workflows. Changes there should still respect the
top-level wrapper contracts.

Contrib estimators under `econirl.contrib` may stay importable without getting
public RTD pages. If a contrib estimator becomes public, create or update its
`internal_docs/estimators/<slug>/` folder first, then add validation evidence,
then expose the concise RTD page.

The migration should not rename public imports unless a separate API migration
plan exists.
