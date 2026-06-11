# Validation Protocol

Runnable validation scripts live in `validation/estimators/<estimator>/`.

Machine-readable validation results live in `validation/results/`.

Validation result files should be strict JSON. They should avoid NaN and
Infinity. If a source computation produces a non-finite value, the public
payload should encode it as `null` or as a documented diagnostic string.

Each result should identify the estimator, the supported target, the primary
cell or data source, the result payload, and whether the evidence is support,
diagnostic only, local smoke evidence, or archived evidence.

Public RTD pages should link to result JSON and runnable scripts in
`validation/`. Internal docs may additionally explain derivations and failure
modes.
