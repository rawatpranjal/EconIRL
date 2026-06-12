# Under the Hood

NFXP nests a dynamic programming problem inside a likelihood problem. The
inner loop solves the Bellman fixed point for a candidate reward parameter;
the outer loop chooses the parameter that maximizes observed choice likelihood.

## Model

The observed data are state, action, and next-state trajectories $(s_{it}, a_{it}, s_{i,t+1})$.

The flow payoff is linear in known features:

$$
u_\theta(s, a) = \phi(s, a)^\top \theta.
$$

The transition kernel is the probability of the next state given the current
state and action:

$$
P_a(s, s') = \Pr(s_{t+1} = s' \mid s_t = s, a_t = a).
$$

Transitions are stored in the package in $(A, S, S)$ orientation. With
discount factor $\beta$ and logit shock scale $\sigma$, the integrated value
function solves the soft Bellman fixed point:

$$
V_\theta(s)
= \sigma \log \sum_a \exp\!\left(
    \frac{u_\theta(s, a) + \beta \sum_{s'} P_a(s, s') V_\theta(s')}{\sigma}
\right).
$$

The choice-specific value is:

$$
Q_\theta(s, a) = u_\theta(s, a) + \beta \sum_{s'} P_a(s, s') V_\theta(s').
$$

The implied conditional choice probability follows the logit rule:

$$
\pi_\theta(a \mid s) =
\frac{\exp(Q_\theta(s, a) / \sigma)}
     {\sum_b \exp(Q_\theta(s, b) / \sigma)}.
$$

## Objective

NFXP maximizes the conditional log likelihood:

$$
\hat{\theta}
= \arg\max_\theta \sum_{i,t} \log \pi_\theta(a_{it} \mid s_{it}).
$$

## Score

The gradient of the log likelihood differentiates through the fixed point.
For observation $i$, the score has the logit form:

$$
\psi_i(\theta)
=
\frac{1}{\sigma}
\left[
    \frac{\partial Q_\theta(s_i, a_i)}{\partial \theta}
    -
    \sum_a \pi_\theta(a \mid s_i)
    \frac{\partial Q_\theta(s_i, a)}{\partial \theta}
\right].
$$

The Q-gradient propagates through the value function. The total derivative
solves:

$$
(I - \beta P_\pi)\frac{\partial V}{\partial \theta}
= \sum_a \pi_\theta(a \mid s)\,\phi(s, a),
$$

where $P_\pi = \sum_a \operatorname{diag}(\pi_\theta(\cdot, a)) P_a$ is the
policy-weighted transition matrix.

## Pseudocode

```text
Input: panel, reward features, transitions, discount beta, shock scale sigma
choose an initial reward parameter vector theta
while the outer optimizer has not stopped:
    form u_theta(s, a) = phi(s, a)' theta
    solve the soft Bellman fixed point  V_theta = T V_theta
    compute Q_theta(s, a) from u_theta, transitions, beta, and V_theta
    compute pi_theta(a | s) by the logit rule
    evaluate sum_{i,t} log pi_theta(a_it | s_it) and its gradient
    pass the likelihood value and gradient to the outer optimizer
return theta, pi_theta, V_theta, standard errors, and diagnostics
```

## Implementation Notes

The package uses a hybrid inner solver: safe successive approximation far from
the fixed point, then Newton-Kantorovich updates near the solution. The outer
optimizer is BHHH by default, which builds a positive semi-definite
approximation to the Hessian from per-observation outer products of scores.
The implementation lives in `econirl.estimation.nfxp`.
