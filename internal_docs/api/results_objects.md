# Results Objects

Result objects should expose parameters, parameter names, convergence status,
estimation time, number of observations, metadata, and diagnostics.

When standard errors are available, result objects should expose standard
errors, variance-covariance information, and confidence intervals in a stable
shape.

Validation JSON should be stricter than interactive results. It must be finite
JSON, should include the estimator name, release or diagnostic status,
simulation cell or data source, metrics, gates when used, and interpretation.
