# Pre-Estimation Checks

Read this page before fitting AIRL. The most important check is whether the
reward object you want is actually state-only, because that is where the
original recovery guarantee applies.

Adversarial training is sensitive to a small number of setup choices. Run these
checks before fitting.

| Check | Why it matters for AIRL |
| --- | --- |
| `reward_arg` vs DGP signal | If the DGP puts its signal in action contrasts, state-only AIRL cannot match it. Set `reward_arg="state_action"` when payoffs are action-dependent. |
| Discriminator loss near log(2) | A final discriminator loss well above log(2) (about 0.693) suggests the game did not converge. Inspect before interpreting reward metrics. |
| `max_rounds` adequate | Too few rounds leave the game undertrained; policy TV stays high. Typical good-fit runs use 150-300 rounds. |
| Feature rank | A rank-deficient reward feature matrix leaves directions of the reward undetermined. |
| Feature condition number | High condition numbers (above ~100) can slow discriminator convergence. |
| State coverage | States not seen in the panel do not contribute discriminator gradients. The policy at those states is only informed by the Bellman propagation from visited neighbors. |
| Action support per state | A state where one action is never taken means the discriminator never gets a gradient signal for that (state, action) pair. |
| Transition row sums | Transitions must be row-stochastic in the `(n_actions, n_states, n_states)` orientation. |

## Simulation Cell Checks

Values from the primary state-only simulation cell (see [Simulation Study](validation.md)):

| Check | Value | Status |
| --- | --- | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 62.58 | pass |
| Observed states | 16 / 16 | pass |
| State-action coverage | 1.000 | pass |
| Minimum action share | 0.242 | pass |
| Final discriminator loss | 1.386 | near log(2) |

The condition number of 62.58 is moderate for a state-only linear reward and
does not impair convergence here, but it is worth noting before applying AIRL
to a different feature basis.

## Common Risk Patterns

**Wrong reward_arg.** When the DGP signal is entirely in action contrasts and
`reward_arg="state"` is used, the reward is averaged across actions before
computing the discriminator logit. Both actions receive equal flow utility; the
policy collapses toward uniform and policy TV stays high regardless of training
length. The fix is `reward_arg="state_action"` plus a longer schedule.

**Undertrained game.** With default `max_rounds=200`, very short runs (below
100 rounds) often leave the discriminator loss above log(2) and the policy
far from the oracle. Increase `max_rounds` and monitor `final_disc_loss`.

**Interpreting raw parameters.** Raw adversarial reward weights are not
structural parameters. Reward is identified only up to potential-based shaping,
so parameter-level comparisons with the data-generating truth are not
meaningful. Policy TV and counterfactual regret are the right scorecards.
