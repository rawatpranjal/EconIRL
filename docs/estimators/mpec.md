# MPEC

Mathematical programming with equilibrium constraints (MPEC) estimates the same
structural dynamic discrete choice model as NFXP. It uses a different numerical
route. NFXP solves the Bellman fixed point inside every likelihood evaluation and
re-solves it each time the reward parameters change. MPEC keeps the value function
as an optimizer variable. It writes the Bellman fixed point as an equality
constraint. The reward parameters and values are then estimated together in one
constrained problem.

Use MPEC when the state space is moderate and the transitions are known or
estimated first. The Bellman residual is then visible directly in the output.
MPEC targets the same tabular structural object as NFXP and CCP.

## The estimation problem

The data are state-action-next-state triples $(s_{it}, a_{it}, s_{i,t+1})$ from a
stationary infinite-horizon dynamic discrete choice model. Flow utility is linear
in known features,

$$
u_\theta(s, a) = \phi(s, a)^\top \theta,
$$

the discount factor $\beta$ and the transition kernels $P_a(s' \mid s)$ are known
or estimated first, and the choice shocks are i.i.d. logit with scale $\sigma$.
Given a value vector $V$, the choice-specific value is the flow payoff plus the
discounted continuation,

$$
Q_\theta(s, a; V) = u_\theta(s, a) + \beta \sum_{s'} P_a(s, s')\, V(s'),
$$

and the implied conditional choice probability is the logit

$$
\pi_{\theta, V}(a \mid s)
= \frac{\exp\!\big(Q_\theta(s, a; V) / \sigma\big)}
       {\sum_b \exp\!\big(Q_\theta(s, b; V) / \sigma\big)}.
$$

The object to maximize is the conditional log-likelihood of the observed choices,
$\sum_{i,t} \log \pi_{\theta, V}(a_{it} \mid s_{it})$. Both estimators rely on the
soft Bellman operator, which maps a value vector to its log-sum-exp,

$$
T_\theta V(s) = \sigma \log \sum_a \exp\!\big(Q_\theta(s, a; V) / \sigma\big).
$$

## From NFXP to MPEC

NFXP treats the value function as a function of the reward parameters. At each
candidate $\theta$ it solves the fixed point $V_\theta = T_\theta V_\theta$ to
convergence. It substitutes $V_\theta$ into the likelihood and passes a single
objective to an unconstrained optimizer. The fixed point is re-solved on every
evaluation.

MPEC (Su and Judd, 2012) embeds that fixed point directly. It keeps $V$ as an
optimization variable alongside $\theta$. It writes the Bellman condition as one
equality constraint per state.

$$
(\hat\theta, \hat V)
= \arg\max_{\theta,\, V} \ \sum_{i,t} \log \pi_{\theta, V}(a_{it} \mid s_{it})
\quad \text{subject to} \quad
V - T_\theta V = 0.
$$

The two estimators are not approximations of each other. At any feasible point the
constraint forces $V = V_\theta$. MPEC and NFXP then evaluate the same likelihood
and target the same maximum likelihood estimate. The difference is the numerical
route, not the structural object. MPEC also reports the final Bellman residual
directly in its output.

## The constrained program

The optimization variable is $x = (\theta, V)$. The constraint has one equality
row per state. SLSQP solves the problem. The objective gradient and the constraint
Jacobian come from JAX, not from finite differences. The value vector starts at the
Bellman fixed point of the initial $\theta$. That gives the optimizer a feasible
starting point and keeps the SQP steps near the constraint surface. No Bellman
fixed point is solved inside the objective. The constraint carries it instead.

## Standard errors

At the optimum the standard errors use the same implicit-score logic as NFXP. The
sensitivity of the value function to the reward parameters solves a linear system.

$$
(I - \beta P_\pi)\, \frac{\partial V}{\partial \theta}
= \sum_a \pi(a \mid s)\, \phi(s, a),
$$

Here $P_\pi$ is the policy-weighted transition matrix. Per-observation scores
follow from this expression, and the robust covariance is their outer product. Two
numbers decide whether a result can be trusted. The first is the final Bellman
constraint violation. The second is whether the standard errors are finite. A high
likelihood value with a violated constraint or a singular information matrix is not
a solution.

## Why it is consistent

The constrained maximum likelihood problem is a sample-average approximation of a
population problem. The reward parameters are the first-stage choice. The Bellman
fixed point is the equilibrium constraint. The log-likelihood is an expectation
estimated from the finite panel. Shapiro and Xu (2005) study this class of
stochastic programs with equilibrium constraints. They show the sample-average
solution is consistent. They show the sample objective converges to its population
counterpart at an exponential rate. They also show that a sharp local optimum of
the population problem is, with probability approaching one, a sharp local optimum
of the sample problem. These results let us treat the MPEC estimate as consistent
and asymptotically normal.

## When it is fragile

The constrained form is clean but not always safe. Iskhakov et al. (2016) show
that the constrained problem can degrade as the state space grows or the discount
factor approaches one. The risky case is not an inner loop diverging. It is the
optimizer reporting success when the constraint violation, the stationarity, or the
standard errors are not credible. Koiso and Otani (2024) add recent evidence from a
sequential-search MPEC estimator. It beat the benchmark in small samples. In larger
samples it had higher bias and error, ran more than four times slower, and
struggled to find local optima. The practical rule is to keep MPEC to moderate
tabular problems. Check the Bellman residual, not just whether the solver reported
success.

## Source Papers

This page draws on {ref}`Su and Judd (2012) <su-judd-2012>` for the constrained
formulation and {ref}`Iskhakov et al. (2016) <iskhakov-2016>` for the comparison
with NFXP. The consistency argument follows
{ref}`Shapiro and Xu (2005) <shapiro-xu-2005>`. The finite-sample cautions come
from {ref}`Koiso and Otani (2024) <koiso-otani-2024>`.

## Quick Decision

| Use MPEC when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete. | The Bellman constraint is too large for a constrained optimizer. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| You need a constrained-likelihood check on NFXP. | You need the fastest repeated comparison run. |
| Bellman constraint diagnostics are central. | You only need a behavioral cloning baseline. |

## Quick Start

```python
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation import simulate_panel

env = RustBusEnvironment(num_mileage_bins=20, discount_factor=0.99)
panel = simulate_panel(env, n_individuals=100, n_periods=50)
utility = LinearUtility.from_environment(env)

model = MPECEstimator(config=MPECConfig(solver="sqp"))
summary = model.estimate(
    panel=panel,
    utility=utility,
    problem=env.problem_spec,
    transitions=env.transition_matrices,
)

print(summary.parameters)
print(summary.metadata["final_constraint_violation"])
```

MPEC's public interface is the lower-level estimator API. See the
[full usage guide](mpec/quick_start.md) and [Under the Hood](mpec/under_the_hood.md)
for the solver path.

## Evidence

MPEC is reported on the low-dimensional action-dependent synthetic
data-generating process. The simulation cell has known rewards, transitions,
policies, values, Q functions, and Type A, Type B, and Type C counterfactual
oracles. The result file records these numbers. MPEC also runs on the
[bus engine](../simulation_studies/rust_bus.md),
[taxi gridworld](../simulation_studies/taxi_gridworld.md), and
[abstract MDP](../simulation_studies/abstract_mdp_1_sanity.md) simulation
study pages alongside the rest of the structural roster. The
[direct optimization study](../simulation_studies/direct_optimization.md)
compares the linear MPEC against its neural sibling and a model-free baseline. It
also checks how stable the estimate is across optimizer starts.

| Evidence | Current state |
| --- | --- |
| Scope | Synthetic constrained-likelihood simulation. |
| Primary cell | `canonical_low_action`. |
| Result file | [mpec_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mpec.json). |
| Bellman constraint check | Reported final violation `7.72e-12`. |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Public example | Uses `MPECEstimator` with the tabular DDC lower-level API. |

## MPEC Guide

- [Context](mpec/context.md)
- [Quick Start](mpec/quick_start.md)
- [Under the Hood](mpec/under_the_hood.md)
- [Pre-Estimation Checks](mpec/pre_estimation.md)
- [Simulation Study](mpec/validation.md)
- [Counterfactuals](mpec/counterfactuals.md)
- [Rust Bus Engine Example](mpec/rust_bus.md)

```{toctree}
:hidden:

mpec/context
mpec/quick_start
mpec/under_the_hood
mpec/pre_estimation
mpec/validation
mpec/counterfactuals
mpec/rust_bus
```
