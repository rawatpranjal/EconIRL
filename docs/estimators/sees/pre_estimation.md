# Pre-Estimation Checks

SEES can fail for reasons that are visible before optimization starts. Run
these checks before treating a SEES result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of reward parameters means theta is not identified. |
| Feature condition number | A high condition number signals unstable reward estimates. |
| Transition row sums | SEES needs valid transition probabilities for continuation values. |
| State coverage | Unobserved states weaken the value-function fit. |
| State-action coverage | Sparse action support weakens both likelihood and Bellman terms. |
| Reward normalization | Reward level and scale need a valid anchor. |
| Basis rank | The sieve must have enough rank to represent the value function. |
| Bellman residual | A small likelihood improvement is not enough if the Bellman penalty is loose. |

## Certified Checks

These rows come from the generated SEES artifact. See
[Validation](validation.md) for the generator script, rendered table source,
and machine-readable JSON.

| Check | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Feature rank | 4 / 4 | 32 / 32 |
| Feature condition number | 4.512 | 1.377 |
| Transition row error | 2.42e-8 | 2.42e-8 |
| Observed states | 21 / 21 | 81 / 81 |
| State-action coverage | 1.000 | 0.959 |
| Minimum action share | 0.325 | 0.281 |
| Basis source | `state_index` | `encoded_state` |
| Basis dimension | 21 | 81 |
| Penalty weight | 100 | 10000 |
| Bellman violation | 5.83e-5 | 3.08e-6 |

The primary cell is the high-dimensional encoded-state DGP. It checks that the
state-feature basis path still recovers known reward, policy, value, Q, and
counterfactual objects.

## Common Risk Patterns

A low-rank reward matrix can make several reward parameters observationally
equivalent. A basis that is too small can fit the likelihood while leaving a
large Bellman residual. Very small penalty weights can produce good in-sample
choice probabilities without a credible structural value function. Wrong
transition orientation can produce plausible arrays and wrong continuation
values.
