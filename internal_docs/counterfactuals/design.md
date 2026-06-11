# Counterfactual Design

Counterfactual support is part of the package identity. It should be treated as
post-estimation infrastructure rather than estimator-specific decoration.

A counterfactual should state what changes, what remains fixed, which policy or
value object is recomputed, and which result fields are comparable to the
baseline.

Validation should separate policy fit from counterfactual validity. A low
policy distance alone is not sufficient when reward, value, Q, or support gates
fail.
