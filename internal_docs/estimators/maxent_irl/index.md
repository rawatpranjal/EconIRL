# MaxEnt IRL Internal Notes

MaxEnt IRL is the legacy trajectory-entropy inverse reinforcement learning
baseline associated with maximum-entropy feature matching. In this repository
it is retained as a package/API surface and diagnostic comparison, but it is not
the preferred structural estimator for dynamic discrete choice comparisons.
MCE-IRL is the causal entropy estimator used when the package needs a
soft-Bellman, time-consistent policy model.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD page: none currently.
- Package wrapper: `../../../src/econirl/estimators/maxent_irl.py`.
- Contrib implementation: `../../../src/econirl/contrib/maxent_irl.py`.
- Validation runner: `../../../validation/estimators/maxent_irl/run.py`.
- Validation result: `../../../validation/results/maxent_irl.json`.

## Objective

The original trajectory MaxEnt model assigns probability to complete
trajectories according to cumulative reward:

```text
P_theta(tau) = exp(theta' f(tau)) / Z(theta)
```

where

```text
f(tau) = sum_t phi(s_t, a_t).
```

The feature-matching gradient is

```text
grad_theta L(theta)
  = E_E[f(tau)] - E_theta[f(tau)].
```

The estimator therefore searches for rewards whose induced trajectory
distribution matches expert feature counts while preserving high entropy among
otherwise equivalent trajectories.

## Difference From MCE-IRL

The important package distinction is not "entropy" versus "no entropy." Both
MaxEnt and MCE use entropy-regularized reasoning. The distinction is what the
entropy is over.

| Dimension | MaxEnt IRL | MCE-IRL |
| --- | --- | --- |
| Entropy object | Whole trajectory distribution | Causal policy at each state/history |
| Policy timing | Can behave as if future path information is available | Conditions only on current information and continuation values |
| Dynamic programming | Less aligned with structural DDC timing | Soft Bellman recursion |
| Package role | Diagnostic/contrib baseline | Preferred entropy IRL estimator |
| Counterfactual interpretation | Limited for structural DDC | Valid under fitted MDP assumptions |

This is why public docs should not present MaxEnt IRL as the main package route
for structural counterfactual analysis. It can still be useful as a baseline,
comparison object, or legacy API surface.

## Current Diagnostic Evidence

The current validation run compares non-causal MaxEnt IRL, causal MCE-IRL, and
behavioral cloning on a 5x5 gridworld with action-dependent features. The script
uses `N_INDIVIDUALS = 100`, `N_PERIODS = 30`, `SEED = 42`, and `beta = 0.95`.

| Estimator | Log likelihood | Time (s) | Parameter cosine | Policy accuracy |
| --- | ---: | ---: | ---: | ---: |
| MaxEnt IRL | -3965.246 | 13.275 | 0.659 | 96.0% |
| MCE-IRL | -3965.096 | 3.230 | 0.682 | 96.0% |
| Behavioral cloning | -3950.154 | 0.269 | n/a | 76.0% |

Interpretation:

- MaxEnt and MCE recover similar greedy policy accuracy in this diagnostic.
- MCE is faster in the current artifact and has slightly better parameter
  cosine.
- Behavioral cloning has lower policy accuracy and no structural reward object.
- This artifact is a comparison receipt, not a release certificate for
  structural MaxEnt counterfactuals.

## Identification And Failure Modes

MaxEnt IRL shares the usual reward-shaping and feature-support issues and adds
the timing concern above. Treat the following as hard review points:

- Does the environment make trajectory entropy a defensible modeling choice?
- Are trajectory feature counts informative after normalization?
- Are deterministic transitions creating degenerate path probabilities?
- Are demonstrations long enough to support trajectory-level feature counts?
- Are results being interpreted as behavior matching rather than structural
  counterfactual recovery?

If the goal is structural dynamic discrete choice comparison, move to MCE-IRL
unless there is a deliberate reason to keep the trajectory MaxEnt baseline.

## Debugging Order

1. Verify state/action indexing and feature construction.
2. Confirm trajectories are built from the same transition system used by the
   estimator.
3. Check feature count scaling and reward normalization.
4. Compare policy accuracy and feature residuals before interpreting theta.
5. Compare against MCE-IRL on the same environment.
6. Avoid counterfactual claims unless the modeling choice has been justified.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/maxent_irl.py`.
- Contrib implementation: `../../../src/econirl/contrib/maxent_irl.py`.
- Validation runner: `../../../validation/estimators/maxent_irl/run.py`.
- Validation JSON: `../../../validation/results/maxent_irl.json`.

## Public Documentation Boundary

There is no current public RTD estimator page for MaxEnt IRL. If one is added,
it should be framed as a legacy or diagnostic baseline and should explicitly
point users toward MCE-IRL for causal entropy dynamic choice work.
