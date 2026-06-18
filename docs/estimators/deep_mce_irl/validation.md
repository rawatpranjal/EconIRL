# Simulation Study

Deep MCE-IRL runs on a synthetic cell with a fixed nonlinear neural reward,
known stochastic transitions, linear state features, and an anchor action that
normalizes the reward. The cell has 32 states, 3 actions, and full
state-action coverage, so every recovery claim is checked against the oracle
reward matrix, policy, value function, Q function, and counterfactual objects.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/run.py).
It writes the results file
[`deep_mce_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/deep_mce_irl/run.py
```

The primary cell results contain no parameter vector. That is by design: a
neural reward map does not have a unique structural parameter vector that can
be compared across networks, so reward-map recovery and behavioral metrics are
the right scorecard. The support cells (`deep_mce_neural_features` and
`deep_mce_neural_reward_features`) exercise the projected theta path, but
projection quality is contingent on the projection being identified.

## Evidence

Deep MCE-IRL is compared against the full structural and IRL rosters on the
[bus engine](../../simulation_studies/rust_bus.md) and
[taxi gridworld](../../simulation_studies/taxi_gridworld.md) pages. See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
