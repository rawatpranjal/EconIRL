# f-IRL Internal Notes

f-IRL learns rewards by matching expert and model marginal distributions under
an f-divergence. The package implementation supports tabular state or
state-action reward scopes, but the current validation should be read narrowly:
the paper-faithful state-marginal, state-only reward cell passes; the
action-dependent DDC cell is a diagnostic negative control.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD source: `../../../docs/estimators/f_irl.md`.
- Lower-level estimator: `../../../src/econirl/estimation/f_irl.py`.
- Validation runner: `../../../validation/estimators/f_irl/run.py`.
- Validation result: `../../../validation/results/f_irl.json`.

## Objective

The paper objective minimizes an f-divergence between expert and policy state
marginals:

```text
min_theta D_f(rho_E(s) || rho_pi_theta(s)).
```

The package computes the soft-optimal policy under the current reward, computes
the induced occupancy, evaluates the configured divergence, and differentiates
through the tabular occupancy machinery.

Supported divergence families in package context include:

- forward KL;
- reverse KL;
- Jensen-Shannon;
- chi-squared;
- total variation where supported by the estimator configuration.

## Marginal Scope

The marginal space is the key interpretation variable.

| Scope | What is matched | Structural interpretation |
| --- | --- | --- |
| State marginal | Expert and model state density | Matches the original f-IRL paper contract. |
| State-action marginal | Expert and model state-action occupancy | More DDC-like, but current action-dependent validation is a negative control. |

State-marginal success does not automatically imply action-dependent reward
recovery. If the public docs discuss action-dependent DDC use, they must cite
the negative-control result.

## Validation Status

Current artifact status:
`paper_state_marginal_pass_action_dependent_diagnostic`.

| Cell | Role | Scope | Converged | Occupancy/state marginal L1 | Reward range | Reward NRMSE | Policy TV | Value NRMSE | Q NRMSE | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `f_irl_paper_state_marginal` | primary paper-faithful cell | state | true | 0.000260 | 1.292 | 0.199 | 0.0121 | 0.130 | 0.100 | pass |
| `canonical_low_action` | action-dependent DDC diagnostic | state-action | true | 0.1609 | 0.000 | 1.000 | 0.141 | 0.832 | 1.353 | negative control |

Primary cell support diagnostics:

- states observed: 8/8;
- state-action coverage: 1.000;
- feature rank: 3/3;
- condition number: 18.471;
- minimum action share: 0.320;
- anchor valid: true.

The primary cell passes convergence, state marginal L1, reward range, reward,
policy, value, Q, and three counterfactual regret gates. The action-dependent
diagnostic fails the reward-range gate with a zero reward range and should not
be treated as structural evidence.

## Counterfactual Evidence

Primary state-marginal cell:

| Counterfactual | Policy TV | Regret | Value RMSE |
| --- | ---: | ---: | ---: |
| Type A | 0.0102 | 0.00708 | 0.00710 |
| Type B | 0.0151 | 0.0124 | 0.0124 |
| Type C | 0.00770 | 0.00273 | 0.00277 |

These are valid only in the paper-faithful validation scope. They do not turn
the failed action-dependent cell into a certified structural estimator.

## Debugging Order

1. Confirm the chosen divergence and marginal space.
2. Check whether reward scope is state or state-action.
3. Verify occupancy computation and transition alignment.
4. Inspect state marginal or occupancy L1 before reward metrics.
5. Check reward range to catch flat-reward failures.
6. Compare policy, value, Q, and counterfactuals only after the marginal and
   reward-range gates pass.

## Implementation Paths

- Lower-level estimator: `../../../src/econirl/estimation/f_irl.py`.
- Validation runner: `../../../validation/estimators/f_irl/run.py`.
- Validation JSON: `../../../validation/results/f_irl.json`.
- Public docs: `../../../docs/estimators/f_irl.md`.

## Public Documentation Boundary

Public RTD should present f-IRL as a state-marginal matching estimator with a
passing paper-faithful validation cell and a clear warning that action-dependent
structural DDC recovery remains diagnostic under current evidence.
