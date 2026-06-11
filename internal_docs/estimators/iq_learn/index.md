# IQ-Learn Internal Notes

IQ-Learn learns an implicit Q function whose soft advantage matches expert
actions while a divergence penalty regularizes the Bellman-implied reward. In
the package it is a useful bridge between behavioral cloning and
Bellman-aware IRL, but the current validation is not strict
counterfactual-certified. Low policy distance and low perturbation regret do
not replace reward, value, Q, and support gates.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD source: `../../../docs/estimators/iq_learn.md`.
- Lower-level estimator: `../../../src/econirl/estimation/iq_learn.py`.
- Validation runner: `../../../validation/estimators/iq_learn/run.py`.
- Sparse-support guard: `../../../validation/estimators/iq_learn/sparse_support_guard.py`.
- Validation result: `../../../validation/results/iq_learn.json`.
- Sparse-support result: `../../../validation/results/iq_learn_sparse_support_guard.json`.

## Objective

For a Q table or Q network, define

```text
V(s) = sigma log sum_a exp(Q(s, a) / sigma)
pi(a | s) = exp((Q(s, a) - V(s)) / sigma)
td_Q(s, a) = Q(s, a) - beta E[V(s') | s, a].
```

The temporal difference `td_Q` is the reward implied by Q through the inverse
Bellman operator. The chi-squared IQ-Learn objective can be read as a penalized
conditional log-likelihood:

```text
sum_i log pi(a_i | s_i)
  - (1 / (4 alpha sigma)) sum_i td_Q(s_i, a_i)^2.
```

The first term behaves like behavioral cloning on Q-induced logits. The second
term encourages smaller implied rewards on expert support. It does not impose a
global Bellman fixed point as a hard structural constraint.

## Identification Boundary

IQ-Learn does not solve reward-shaping non-identification. The regularizer
selects one small-implied-reward representative on expert support; that
representative is not automatically an economically anchored reward. For
structural counterfactual use, support and Bellman-object gates must pass.

The public and internal docs should keep these objects separate:

- imitation policy under the observed environment;
- Bellman-implied reward;
- projected reward;
- value and Q recovery;
- state and state-action support;
- counterfactual regret after structural gates are satisfied.

## Validation Status

The current artifact status is `strict_structural_counterfactual_fail` with
`counterfactual_valid_certified = false`.

| Cell | Q type | Coverage | Policy TV | Raw reward NRMSE | Projected reward NRMSE | Value NRMSE | Q NRMSE | Main failed gates |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `canonical_low_action` | tabular | state 1.000, state-action 1.000 | 0.0407 | 0.381 | 0.277 | 0.486 | 0.481 | reward, value, Q |
| `canonical_high_action` | neural | state 1.000, state-action 0.959 | 0.0693 | 0.997 | 0.786 | 1.753 | 1.422 | policy, reward, value, Q, A/B regret |
| `canonical_low_state_only` | tabular | state 1.000, state-action 1.000 | 0.0366 | 0.750 | 0.281 | 0.570 | 0.552 | reward, value, Q, B regret |

The primary cell is `canonical_low_action`. It converges in 173 iterations on
160000 observations, with `alpha = 1.0`, `divergence = chi2`, and a tabular Q
head. It passes support, convergence, policy TV, and regret gates, but fails
raw Bellman reward, projected reward, value, and Q gates.

## Sparse-Support Guard

The sparse-support guard is an important non-release artifact. It deliberately
uses a tiny panel with only one observed state and one observed state-action
pair:

```text
state coverage = 1 / 3 = 0.333
state-action coverage = 1 / 6 = 0.167
```

All non-support fixture metrics are set to pass, yet the artifact is still not
counterfactual-valid because support gates fail. This guard prevents a future
release from treating small policy or regret numbers as sufficient evidence
when the expert panel does not cover the relevant state-action space.

## Debugging Order

1. Check expert state coverage and state-action coverage first.
2. Verify transition rows, discount, and scale.
3. Confirm Q type (`tabular` versus `neural`) and divergence.
4. Inspect policy TV and log-likelihood.
5. Compute Bellman-implied reward on and off expert support.
6. Compare projected reward, value, and Q.
7. Treat counterfactual regret as meaningful only after support and structural
   gates pass.

## Implementation Paths

- Lower-level estimator: `../../../src/econirl/estimation/iq_learn.py`.
- Validation runner: `../../../validation/estimators/iq_learn/run.py`.
- Sparse-support guard: `../../../validation/estimators/iq_learn/sparse_support_guard.py`.
- Validation JSON: `../../../validation/results/iq_learn.json`.
- Sparse-support JSON: `../../../validation/results/iq_learn_sparse_support_guard.json`.
- Public docs: `../../../docs/estimators/iq_learn.md`.

## Public Documentation Boundary

Public RTD should describe IQ-Learn as a useful implicit-Q imitation and IRL
estimator with strict support requirements. It should not imply
counterfactual-valid reward recovery under the current artifacts.
