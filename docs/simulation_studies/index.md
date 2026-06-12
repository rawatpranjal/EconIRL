# Simulation Studies

Simulation studies run the estimators on synthetic data where the reward, the
transitions, and the true-parameter policy are all known. Because the
data-generating process is set by hand, we can measure exactly how well each
estimator recovers what produced the data, and where it fails.

Each sub-page below is one concrete experiment. They are ordered as a teaching
arc, from a canonical replication through a sanity check to regimes large
enough that only approximation-based estimators remain feasible.

| Page | Environment | Size | Roster focus | What it demonstrates |
| --- | --- | --- | --- | --- |
| [Bus engine replacement](rust_bus.md) | Keep-or-replace mileage model (Rust 1987). | 20 states x 2 actions | Everything that runs. | The canonical benchmark: who recovers the cost parameters, at what compute cost. |
| [Gridworld navigation](taxi_gridworld.md) | Walk to an absorbing goal on a grid. | 64 states x 5 actions | IRL-weighted, structural contrast. | Thin state coverage: what happens where the data rarely goes. |
| [Abstract MDP 1: sanity](abstract_mdp_1_sanity.md) | Small random MDP, linear reward. | 8 states x 2 actions | Everything that runs. | The sanity check: an easy problem every correct estimator must pass. |
| [Abstract MDP 2: harder](abstract_mdp_2_harder.md) | The same generator, hardened three ways. | 300 states; 24-state collinear cell | Structural family. | Runtime at scale, inference quality near discount one, and broken identification. |
| [Abstract MDP 3: high dimension](abstract_mdp_3_highdim.md) | The same generator at benchmark scale. | 3000 states x 2 actions | Approximation-based estimators. | Where classical solvers stop being practical, measured by feasibility probes. |
| [Direct optimization](direct_optimization.md) | Value-lifted estimation under correct and misspecified rewards. | varies | MPEC, neural MPEC, GLADIUS. | How the value-lifted family degrades under reward misspecification. |

A through-line runs across them. Behavioral recovery is usually robust: most
estimators reproduce the choice probabilities well. The methods separate
elsewhere, on whether the underlying parameters are identified, on whether
their recovered model transfers to a changed world, on cost, and, as the state
space grows, on whether an estimator can run at all.

## How to read the tables

Every sub-page reports the same honest objects: a results table at top with
the exact recovered parameters, policy distance, and counterfactual regret;
the exact code snippet used for each estimator; and a plain reading of what
worked, what did not, and why. Numbers are rendered from a saved results file
so they never drift from the prose. Failures, crashes, and budget timeouts
stay in the table with their verbatim messages.

Counterfactual regret is the welfare lost when an estimator's recovered model
is carried into a changed world, following the package's three intervention
families: Type A shifts a payoff component, Type B changes the transition
dynamics, and Type C penalizes an action away. Estimators that recovered a
reward in the linear feature gauge re-solve it under each intervention and
adapt; estimators that only matched behavior keep their frozen policy, which
is exactly why their Type C regret is large. Parameter recovery is reported
only for the structural family, which estimates the parameters in the same
gauge as the truth; reward parameters from the IRL family live in a different
gauge (reward is only partially identified from behavior), so parameter-level
comparisons across that divide would be meaningless.

The estimators themselves are documented in the
[estimator catalog](../estimators.md).

```{toctree}
:maxdepth: 1

rust_bus
taxi_gridworld
abstract_mdp_1_sanity
abstract_mdp_2_harder
abstract_mdp_3_highdim
direct_optimization
```
