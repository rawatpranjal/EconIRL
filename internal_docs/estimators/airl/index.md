# AIRL Internal Notes

Adversarial inverse reinforcement learning structures a discriminator so that
one component can represent transferable reward and another can absorb
potential-based shaping. In this package, AIRL is useful as an adversarial IRL
comparison and as a bridge to AIRL-Het, but its public claims must remain
narrow: the current validation passes under the original state-only
identification conditions and fails the anchored action-dependent diagnostic.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD source: `../../../docs/estimators/airl.md`.
- Neural wrapper: `../../../src/econirl/estimators/neural_airl.py`.
- Lower-level estimator: `../../../src/econirl/estimation/adversarial/airl.py`.
- Validation runner: `../../../validation/estimators/airl/run.py`.
- Validation result: `../../../validation/results/airl.json`.

## Model

AIRL decomposes the discriminator logit as

```text
f_theta_phi(s, a, s') = g_theta(s) + beta h_phi(s') - h_phi(s)
```

and defines

```text
D(s, a, s') = exp(f_theta_phi(s, a, s'))
              / (exp(f_theta_phi(s, a, s')) + pi(a | s)).
```

The reward signal used for policy learning is the discriminator log-odds,
commonly written as

```text
log D - log(1 - D) = f_theta_phi(s, a, s') - log pi(a | s).
```

The theoretical reward-recovery result requires a state-only `g_theta(s)`, a
decomposable MDP, and convergence of the adversarial game. The shaping network
`h_phi` is supposed to absorb value-like terms that otherwise contaminate the
reward.

## Identification Boundary

The core ambiguity is potential-based shaping:

```text
r'(s, a, s') = r(s, a, s') + beta h(s') - h(s).
```

This transformation preserves optimal behavior under the original dynamics but
does not generally preserve counterfactual behavior under new dynamics. AIRL's
structured discriminator is designed to separate `g_theta` from `h_phi`, but
the original guarantee is state-only. Dynamic discrete choice settings often
have action-dependent payoffs, exit actions, and absorbing states, which is why
the anchored heterogeneous extension is documented separately under
`../airl_het/index.md`.

## Validation Status

The current artifact status is `partial`. AIRL passes the paper-style
state-only identification cell and fails the anchored action-dependent
diagnostic.

| Cell | Role | Converged | Reward NRMSE | Policy TV | Value NRMSE | Q NRMSE | Counterfactual regrets | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `airl_paper_identification` | primary state-only cell | true | 0.0998 | 0.00596 | 0.1099 | 0.1201 | A 0.00292, B 0.00381, C 0.00496 | pass |
| `airl_anchor_action_dependent` | negative action-dependent diagnostic | false | 1.1606 | 0.4030 | 1.9954 | 1.2601 | A 10.500, B 13.973, C 4.405 | fail |

Primary cell details:

- observations: 24000;
- states observed: 16/16;
- state-action coverage: 1.000;
- feature rank: 4/4;
- condition number: 62.575;
- final discriminator loss: 1.386;
- training rounds: 150;
- reward argument: state;
- learned shaping: true.

The primary cell passes all current gates, including reward, policy, value, Q,
and three counterfactual regret gates. The action-dependent cell fails every
hard gate and should remain a diagnostic warning, not a public success claim.

## Interpretation For Maintainers

Use AIRL claims in this order:

1. State-only reward recovery under the paper assumptions.
2. Behavior matching in the validation environment.
3. Transfer or counterfactual discussion only when the reward component is
   plausibly disentangled from shaping.
4. For action-dependent structural DDC claims, route to AIRL-Het or another
   anchored estimator.

The low parameter cosine in the primary state-only cell is not the release
target by itself because the primary validation focuses on reward map, policy,
value, Q, and counterfactual behavior. Do not use raw adversarial weights as
structural parameters.

## Debugging Order

1. Verify the discriminator input tuple `(s, a, s')`.
2. Confirm whether `g_theta` is state-only or state-action.
3. Check decomposability, absorbing states, and any anchor action.
4. Inspect discriminator loss before interpreting reward metrics.
5. Compare reward under a fixed gauge.
6. Compare policy, value, Q, and counterfactual metrics.
7. If action-dependent recovery fails, do not tune around the failure until the
   identification design is reconsidered.

## Implementation Paths

- Neural wrapper: `../../../src/econirl/estimators/neural_airl.py`.
- Lower-level estimator: `../../../src/econirl/estimation/adversarial/airl.py`.
- Validation runner: `../../../validation/estimators/airl/run.py`.
- Validation JSON: `../../../validation/results/airl.json`.
- Public docs: `../../../docs/estimators/airl.md`.

## Public Documentation Boundary

Public RTD should describe AIRL as a structured adversarial IRL estimator, show
the state-only validation receipt, and explicitly avoid action-dependent
counterfactual claims unless the page routes to AIRL-Het.
