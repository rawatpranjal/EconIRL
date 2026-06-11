# Behavioral Cloning Internal Notes

Behavioral cloning is a supervised imitation baseline. It estimates
`P(a | s)` directly from demonstration frequencies, optionally with Laplace
smoothing. It does not solve a Bellman equation, estimate primitive rewards, or
support structural counterfactuals.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD page: none currently.
- Lower-level estimator: `../../../src/econirl/estimation/behavioral_cloning.py`.
- Tests: `../../../tests/test_behavioral_cloning.py`.
- Validation result: none currently.

## Estimator Contract

The estimator counts state-action observations:

```text
count(s, a) = number of demonstrations choosing action a in state s
```

with optional smoothing:

```text
P_hat(a | s) = (count(s, a) + smoothing)
             / sum_b (count(s, b) + smoothing).
```

It returns the flattened policy as parameters, a zero value function, and
standard errors set to `nan` because frequency-count policy imitation is not a
structural parameter estimator.

## Interpretation

Behavioral cloning is useful as:

- a lower-bound imitation baseline;
- a diagnostic for state-action support;
- a comparison point for methods that claim to use dynamic structure.

It is not useful as:

- reward recovery;
- welfare analysis;
- counterfactual policy simulation under changed transitions;
- structural utility estimation.

Any estimator that cannot beat behavioral cloning on policy imitation is not
extracting useful dynamic structure from the MDP.

## Failure Modes

- With `smoothing = 0`, unvisited states keep all-zero policy rows and emit a
  warning.
- With smoothing, unvisited states receive fallback action probabilities that
  may hide support failure.
- High policy accuracy can coexist with no reward or value recovery.
- It can perform well in easy observed-policy prediction while failing
  out-of-distribution states.

## Debugging Order

1. Check state coverage and state-action coverage.
2. Inspect smoothing value.
3. Look for warnings about unvisited states.
4. Compare policy rows against empirical CCPs.
5. Use only as a baseline against structural or IRL estimators.

## Implementation Paths

- Lower-level estimator: `../../../src/econirl/estimation/behavioral_cloning.py`.
- Tests: `../../../tests/test_behavioral_cloning.py`.

## Public Documentation Boundary

Do not add a standalone public estimator page unless the docs need a baseline
page. If included publicly, label it as non-structural.
