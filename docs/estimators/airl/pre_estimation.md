# Pre-Estimation Checks

## Important Links

- [AIRL Overview](../airl.md)
- [Identification Boundary](identification.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)

Stop before fitting if the reward is not state-only. More adversarial rounds do
not repair a reward specification that puts its signal in action contrasts.

## Required checks

| Check | Required condition |
| --- | --- |
| Reward scope | Features depend on state, not action or context. |
| Feature rank | Full column rank over the declared state feature matrix. |
| Feature condition | Finite and not dominated by nearly duplicate columns. |
| State coverage | Every state used for interpretation appears in the panel. |
| Action support | Every declared action appears. Thin state-action cells are reported. |
| Transitions | Finite, nonnegative, and row-stochastic. |
| Orientation | `(n_actions, n_states, n_states)`. |
| Decomposability | Supported by the problem design and scientific argument. |

The estimator checks coded support, transition shape, row sums, and state
feature rank before optimization. Decomposability is a property of the MDP. It
cannot be established by optimizer success alone.

## Fit checks

After fitting, inspect the optimization block in `diagnostics_`.

- Confirm the fit reached the policy-change stopping rule.
- Inspect the final discriminator loss in `diagnostics_`. The full loss path is
  available in `result_.metadata["disc_losses"]`.
- Compare predicted and observed actions on held-out trajectories.
- Check that the recovered policy changes when a material transition change is
  introduced.

A solver completion flag does not establish reward recovery. The controlled
study grades normalized reward error, policy distance, value error, Q error,
and changed-dynamics behavior together.

## Fail-closed inputs

Action-dependent `RewardSpec` objects raise before training. `context=` also
raises before training. Both errors point to AIRL-Het because they require a
different identification design.
