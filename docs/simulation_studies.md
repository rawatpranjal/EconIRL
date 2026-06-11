# Simulation Studies

Simulation studies run estimators on synthetic data where the reward,
transitions, and true-parameter policy are all known. Because the data-generating
process is set by hand, we can measure exactly how well each estimator recovers
what produced the data. This page compares estimators across a ladder of
synthetic problems, from a canonical recoverable case to regimes designed to
separate methods.

## Build your own data-generating process

Two generators produce the synthetic MDPs. `random_mdp` builds an abstract
problem from a few knobs (states, actions, branching factor); `ArrayMDP` takes a
transition tensor, a feature tensor, and reward parameters that you supply
directly.

```python
from econirl.environments import random_mdp, ArrayMDP
from econirl.simulation import simulate_panel

# An abstract MDP with sparse transitions and linear reward.
env = random_mdp(num_states=30, num_actions=2, branching=4, discount_factor=0.95, seed=0)
panel = simulate_panel(env, n_individuals=500, n_periods=80, seed=1)

# Or inject your own dynamics, features, and reward.
custom = ArrayMDP(transitions, features, theta={"cost": -1.0, "value": 0.5})
```

Both produce a panel that any estimator consumes, and both expose the true
policy and reward, so recovery can be scored.

## The difficulty ladder

Each cell isolates one source of difficulty.

| Cell | Stresses |
|---|---|
| Simple binary (Rust bus) | none; every estimator should recover it |
| Stochastic mid-size | genuinely stochastic transitions, moderate state count |
| Near-unit discount | discount near 1, where continuation values dominate |
| Nonlinear reward | a reward that linear-utility estimators cannot represent |
| Larger state space | a larger state space; checks the methods stay cheap (the dense transition ceiling sits higher, in the low thousands of states) |
| Rank-deficient features | collinear features that break parameter identification |

## How to read the metrics

Estimators are scored on two kinds of metric, kept separate on purpose.

**Behavioral metrics** apply to every estimator. Policy total-variation distance
measures how far the recovered choice probabilities sit from the true-parameter
policy; value RMSE compares value functions. These are comparable across all
methods because the recovered behavior is comparable.

**Parameter metrics** (bias, empirical standard error, and 95% confidence-interval
coverage) apply only to the structural family, which recovers a finite-dimensional
reward in the same gauge as the data-generating process and reports real standard
errors. Reward functions recovered by inverse-reinforcement-learning methods are
only partially identified, so a single parameter-bias number is not comparable
across families ([Skalse et al 2023](https://arxiv.org/abs/2203.07475)). We never
place one on a method that cannot support it.

Coverage uses a normal-approximation interval. Every reported mean carries a Monte
Carlo standard error, so the uncertainty from a finite number of replications is
visible rather than hidden.

Two honesty notes. First, not every estimator returns usable standard errors on
every problem; where an estimator fails to, coverage is reported as not available
rather than guessed, and the failure-mode map flags it. Second, an estimator's own
convergence flag and its behavioral recovery can disagree: MCE-IRL reports its
inner objective as not converged on these cells, yet its recovered policy tracks
the truth closely. Both are reported as they are.

## Results

A theme runs through the numbers below: behavioral recovery is robust. Across
every cell the estimators reproduce the true-parameter policy closely, even when
the reward is nonlinear or the features are collinear. The methods separate
elsewhere, on whether the underlying parameters are identified at all, on whether
confidence intervals are available and cover, and on cost. Read the recovery
figure for the shared behavioral story, then the other three for where the
methods diverge.

```{figure} _static/simulation_studies/recovery_vs_complexity.png
Behavioral recovery, measured by policy total-variation distance, for each
estimator across the difficulty ladder. Lower is better. Recovery stays tight
across the ladder, which is the point: behavior is the robust part.
```

```{figure} _static/simulation_studies/coverage.png
Coverage of nominal 95% confidence intervals for the structural family. The
dashed line marks 0.95; bars carry Monte Carlo error.
```

```{figure} _static/simulation_studies/bias_variance.png
Bias against empirical standard error for the structural family. Points toward
the origin recover the parameters tightly and with little spread.
```

```{figure} _static/simulation_studies/runtime.png
Mean fit time per replication across the ladder, on a log scale. This is where
the methods separate on cost.
```

The tables below report the full per-cell numbers and a map of which
data-generating condition strains which estimator.

```{include} _generated/simulation_studies_results.md
```

## Reproduce

The numbers and figures on this page are produced by one script:

```bash
python scripts/run_simulation_studies.py --replications 20
```

It runs the benchmark over the ladder, writes
`validation/results/simulation_studies.json`, and regenerates the figures and
tables. The generation date, replication count, and package version are stamped
in the results above.
