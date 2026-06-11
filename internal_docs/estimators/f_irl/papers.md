# f-IRL Paper Context

Primary source: Ni et al. (2020) for inverse reinforcement learning via state
marginal matching. Public citations live in `../../../docs/references.md`.
Broader paper routing lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Expert state density | Empirical state marginal | Primary validation follows this scope. |
| Policy state marginal | Soft-optimal occupancy | Computed from the current reward and transition model. |
| f-divergence | Configured divergence family | Forward KL is used in the current primary artifact. |
| Analytic gradient | Autodiff through tabular occupancy | Package uses closed-form tabular occupancy. |
| Stationary reward | State reward map | Primary cell validates state-only reward. |
| State-action extension | Diagnostic package cell | Current action-dependent DDC cell fails reward-range evidence. |

## Derivation Checklist

1. Define expert and model state marginals.
2. State the f-divergence objective.
3. Explain how the reward induces a soft-optimal policy and occupancy.
4. Map divergence derivatives to reward updates.
5. Separate state-marginal recovery from state-action occupancy recovery.
6. Include reward-range checks to catch flat-reward solutions.
7. Preserve the current action-dependent negative-control result.

## Paper Notes To Retain

The paper emphasizes:

- state marginal matching rather than trajectory distribution matching;
- support for multiple f-divergence families;
- learning a stationary reward rather than only a discriminator reward;
- sample efficiency in imitation settings;
- downstream use of the learned reward in the paper's robotics tasks.

In this package, the durable result is narrower: the state-marginal/state-only
validation cell passes, while the action-dependent DDC cell is not structural
release evidence.
