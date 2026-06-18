# Simulation Study

AIRL runs on two synthetic cells. The primary cell is `airl_paper_identification`,
a state-only reward DGP designed to match the original identification conditions
of Fu et al. (2018). The second cell is `airl_anchor_action_dependent`, an
action-dependent DGP with an absorbing state and an anchored exit action. That
cell is a diagnostic run to confirm where the method fails, not a validated use
case.

In both cells, the transitions, reward, policy, value function, Q function, and
Type A, Type B, and Type C counterfactual oracles are chosen before generating
the panel. The estimator sees only the demonstrations, the transitions, and the
reward feature basis. The true objects are held back for evaluation.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/airl/run.py).
It writes the results file
[`airl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl.json).

```bash
PYTHONPATH=src:. python validation/estimators/airl/run.py --quiet-progress
```

The diagnostic cell `airl_anchor_action_dependent` fails all checks. With a
state-only reward, the discriminator cannot represent the action contrast that
drives behavior in an action-dependent DGP. That is a structural failure, not a
tuning problem. The identification page explains the boundary in detail.

## Evidence

AIRL is compared against the full structural and IRL rosters on the
[bus engine](../../simulation_studies/rust_bus.md) and
[taxi gridworld](../../simulation_studies/taxi_gridworld.md) pages. See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
