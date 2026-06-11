# GCL Internal Notes

Guided Cost Learning is retained through contrib and a sklearn-style wrapper.
It learns a neural cost function with trajectory-level importance sampling and
updates the policy with soft value iteration. It is useful as a historical
neural IRL baseline and for integration tests, but it is not part of the
current public RTD estimator surface.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD page: none currently.
- Package wrapper: `../../../src/econirl/estimators/gcl.py`.
- Contrib implementation: `../../../src/econirl/contrib/gcl.py`.
- Compatibility import: `../../../src/econirl/estimation/gcl.py`.
- Validation runner: `../../../validation/estimators/gcl/run.py`.
- Validation result: `../../../validation/results/gcl.json`.

## Algorithm

GCL models cost rather than reward:

```text
reward(s, a) = -cost(s, a).
```

The package implementation uses a neural cost function and alternates:

1. sample trajectories from the current policy;
2. compute importance weights for sampled trajectories;
3. update the cost network so demonstrations have lower cost than samples;
4. convert cost to reward;
5. update the policy with soft value iteration;
6. repeat until the cost loss stabilizes or the iteration cap is reached.

The high-level wrapper exposes fitted `cost_matrix_`, `reward_matrix_`,
`policy_`, and `cost_function_`. The compatibility module exists so older
imports from `econirl.estimation.gcl` still resolve to contrib classes.

## Current Validation Evidence

The tracked artifact compares GCL with MCE-IRL on a 5x5 gridworld:

| Method | Policy accuracy | Time (s) |
| --- | ---: | ---: |
| MCE-IRL | 1.00 | 83.66 |
| GCL | 0.84 | 99.35 |

Environment:

- grid size: 5;
- states: 25;
- actions: 5;
- observations: 10000;
- discount: 0.99.

This is a comparison receipt, not a structural release certificate. GCL has no
current counterfactual-valid validation bundle with reward, value, Q, and
counterfactual gates.

## Import Surface

Use these imports:

```python
from econirl.estimators import GCL
from econirl.contrib.gcl import GCLEstimator, GCLConfig
```

The old compatibility import is supported for moved low-level classes:

```python
from econirl.estimation.gcl import GCLEstimator, GCLConfig
```

Do not document GCL as a production estimator until a public page and stronger
validation artifact exist.

## Debugging Order

1. Verify trajectory construction and state-action encoding.
2. Check neural cost matrix shape and reward sign convention.
3. Inspect importance weights for degeneracy.
4. Compare policy accuracy against MCE-IRL and behavioral cloning.
5. Check whether the compatibility import path is required by old tests.
6. Treat reward values as neural cost outputs, not identified structural
   utility parameters.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/gcl.py`.
- Contrib estimator: `../../../src/econirl/contrib/gcl.py`.
- Compatibility shim: `../../../src/econirl/estimation/gcl.py`.
- Validation runner: `../../../validation/estimators/gcl/run.py`.
- Validation JSON: `../../../validation/results/gcl.json`.

## Public Documentation Boundary

There is no current public RTD estimator page. If one is added, it should be
framed as a contrib neural cost-learning baseline with limited release
evidence.
