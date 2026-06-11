# Estimator Protocol

Public estimators should follow the sklearn-style wrapper contract.

The expected workflow is `fit`, inspect result attributes, call `summary` when
available, and run post-estimation or counterfactual methods when supported.

Low-level estimators may expose richer configuration objects, panel objects,
utility objects, and transition tensors. Public wrappers should hide that
complexity unless the user needs advanced control.

All estimators should make convergence, number of observations, parameter
names, standard errors when available, metadata, and failure reasons visible.
