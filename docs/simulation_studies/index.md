# Simulation Studies

Simulation studies run the estimators on synthetic data where the reward, the
transitions, and the true-parameter policy are all known. Because the
data-generating process is set by hand, we can measure exactly how well each
estimator recovers what produced the data, and where it fails.

Each sub-page below is one concrete experiment. They are ordered as a teaching
arc, from a canonical replication through a sanity check to regimes large enough
that only machine-learning-powered estimators remain feasible.

A through-line runs across them. Behavioral recovery is usually robust: most
estimators reproduce the choice probabilities well. The methods separate
elsewhere, on whether the underlying parameters are identified, on whether their
recovered model transfers to a changed world (counterfactual regret), on cost,
and, as the state space grows, on whether an estimator can run at all.

Every sub-page reports the same honest objects: a results table at top with the
exact recovered parameters, policy distance, and counterfactual regret; the exact
code snippet used for each estimator; and a plain reading of what worked, what did
not, and why. Numbers are rendered from a saved results file so they never drift
from the prose. Counterfactual regret follows the package's Type A (payoff), Type
B (transition), and Type C (action-set) taxonomy.

```{toctree}
:maxdepth: 1

rust_bus
taxi_gridworld
abstract_mdp_1_sanity
abstract_mdp_2_harder
abstract_mdp_3_highdim
cross_estimator_ladder
```
