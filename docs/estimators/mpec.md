# MPEC

Mathematical programming with equilibrium constraints (MPEC) estimates the same
structural dynamic discrete choice model as NFXP, by a different numerical route.
NFXP hides the Bellman fixed point inside every likelihood evaluation, re-solving
for the value function each time the reward parameters move. MPEC instead lifts
the value function into the optimizer as a free variable and imposes the Bellman
fixed point as an explicit equality constraint, so reward parameters and values
are solved for jointly in a single constrained problem.

Use MPEC when the state space is moderate, the transitions are known or
first-stage estimated, and you want the Bellman residual on the table as a
diagnostic rather than buried inside an inner solve. It targets the same tabular
structural object as NFXP and CCP.

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

NFXP treats the value function as a function of $\theta$. At each candidate
$\theta$ it solves the fixed point $V_\theta = T_\theta V_\theta$ to convergence,
substitutes $V_\theta$ into the likelihood, and hands a single objective to an
unconstrained optimizer. The fixed point is real work repeated on every
evaluation.

MPEC ({ref}`Su and Judd, 2012 <su-judd-2012>`) refuses to hide it. It keeps $V$
as an explicit optimization variable alongside $\theta$ and writes the Bellman
condition as a per-state equality constraint:

$$
(\hat\theta, \hat V)
= \arg\max_{\theta,\, V} \ \sum_{i,t} \log \pi_{\theta, V}(a_{it} \mid s_{it})
\quad \text{subject to} \quad
V - T_\theta V = 0.
$$

The two estimators are not approximations of each other. At any feasible point the
constraint forces $V = V_\theta$, so MPEC and NFXP evaluate the identical dynamic
discrete choice likelihood and target the same maximum likelihood estimate. The
difference is numerical geometry, not the structural object. The one visible
payoff is diagnostic: the fitted summary reports the final Bellman residual
directly, where NFXP dissolves it inside each inner solve.

## The constrained program

The joint optimization variable is $x = (\theta, V)$, with one equality-constraint
row per state. The implementation solves it with SLSQP, supplying the objective
gradient and the constraint Jacobian from JAX rather than by finite differences.
The value vector is initialized at the Bellman fixed point of the starting
$\theta$, which places the optimizer at a feasible start and lets the SQP steps
stay near the constraint surface. There is no nested Bellman solve inside the
objective; the constraint carries the fixed point instead.

## Standard errors

At the constrained optimum the standard errors follow the same implicit-score
logic as NFXP. The sensitivity of the value function to the reward parameters
solves the linear system

$$
(I - \beta P_\pi)\, \frac{\partial V}{\partial \theta}
= \sum_a \pi(a \mid s)\, \phi(s, a),
$$

where $P_\pi$ is the policy-weighted transition matrix. Per-observation score
contributions follow from this expression, and the robust covariance is their
outer product. A reported run gates on two numbers, not one: the final Bellman
constraint violation and the finiteness of the standard errors. A high likelihood
value with a violated constraint or a singular information matrix is not a
solution.

## Why it is consistent

The constrained maximum likelihood problem is a sample-average approximation of a
population problem: the reward parameters are the first-stage choice, the Bellman
fixed point is the equilibrium constraint, and the log-likelihood is an
expectation estimated from the finite panel.
{ref}`Shapiro and Xu, 2005 <shapiro-xu-2005>` study exactly this class of
stochastic programs with equilibrium constraints and show that the sample-average
solution is consistent, that the sample objective converges to its population
counterpart at an exponential rate, and that a sharp local optimum of the
population problem is, with probability approaching one, a sharp local optimum of
the sample problem. Those results are what license treating the MPEC estimate as
a consistent and asymptotically normal estimator rather than the output of a
solver.

## When it is fragile

The constrained form is clean but not unconditionally safe.
{ref}`Iskhakov et al., 2016 <iskhakov-2016>` document that the constrained problem
can degrade as the state space grows or the discount factor approaches one, and
that the dangerous failure mode is not an inner loop diverging. It is the
optimizer reporting success while the constraint violation, stationarity, or
standard errors are not credible. {ref}`Koiso and Otani, 2024 <koiso-otani-2024>`
add recent evidence from a sequential-search MPEC estimator: it beat the benchmark
at small samples but had higher bias and root-mean-squared error at larger
samples, ran more than four times slower, and struggled to find local optima. The
practical stance is to keep MPEC to moderate tabular problems, gate hard on the
Bellman residual, and read a converged flag as necessary but not sufficient.

## Source Papers

This page draws on {ref}`Su and Judd (2012) <su-judd-2012>` for the constrained
formulation, {ref}`Iskhakov et al. (2016) <iskhakov-2016>` for the computational
critique against NFXP, {ref}`Shapiro and Xu (2005) <shapiro-xu-2005>` for the
sample-average-approximation consistency theory, and
{ref}`Koiso and Otani (2024) <koiso-otani-2024>` for recent finite-sample evidence
on MPEC fragility.

## Quick Decision

| Use MPEC when | Prefer another estimator when |
| --- | --- |
| States and actions are discrete. | The Bellman constraint is too large for a constrained optimizer. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling problem. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| You need a constrained-likelihood check on NFXP. | You need the fastest repeated comparison run. |
| Bellman constraint diagnostics are central. | You only need a behavioral cloning baseline. |

## Minimal Fit

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

MPEC's public surface is the lower-level estimator API; see
[Quick Start](mpec/quick_start.md) for full usage, and
[Under the Hood](mpec/under_the_hood.md) for the solver path and pseudocode.

## Evidence

MPEC is reported on the low-dimensional action-dependent synthetic data-generating process. The
simulation cell has known rewards, transitions, policies, values, Q functions,
and Type A, Type B, and Type C counterfactual oracles. The machine-readable
results file records the reported results. MPEC also runs on the
[bus engine](../simulation_studies/rust_bus.md),
[taxi gridworld](../simulation_studies/taxi_gridworld.md), and
[abstract MDP](../simulation_studies/abstract_mdp_1_sanity.md) simulation
study pages alongside the rest of the structural roster. The
[direct optimization study](../simulation_studies/direct_optimization.md)
compares the linear MPEC against its neural sibling and a model-free baseline,
and includes a multi-start local-optima check motivated by the fragility
literature above.

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic constrained-likelihood simulation. |
| Primary cell | `canonical_low_action`. |
| Machine-readable results file | [mpec_results.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mpec.json). |
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
