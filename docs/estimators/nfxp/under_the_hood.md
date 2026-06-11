# Under the Hood

NFXP nests a dynamic programming problem inside a likelihood problem. The inner
problem solves the Bellman fixed point for a candidate reward parameter. The
outer problem chooses the reward parameter that maximizes observed choice
likelihood.

## Optimization Setup

The observed panel supplies state, action, and next-state records. The
transition law is estimated from the panel or supplied directly. Reward
features, the discount factor, and the logit shock scale are fixed before
optimization.

NFXP optimizes the structural reward parameter vector `theta`. For each
candidate `theta`, the estimator solves the soft Bellman fixed point, forms the
choice-specific values, computes the implied logit policy, and evaluates the
conditional log likelihood of the observed actions. The public simulation path
uses the hybrid Bellman solver inside the likelihood and the configured outer
optimizer for `theta`.

## Pseudocode

```text
Given panel, reward features, transitions, discount, and shock scale
Initialize theta
Repeat until the outer optimizer stops:
    Build reward u_theta(s, a)
    Solve V_theta = T_theta(V_theta)
    Compute Q_theta(s, a) and pi_theta(a | s)
    Evaluate the conditional log likelihood
    Update theta using the likelihood gradient or optimizer step
Return theta, policy, value function, standard errors, and diagnostics
```

## Model

The observed data are state, action, and next-state trajectories.

$$
(s_{it}, a_{it}, s_{i,t+1})
$$

The flow payoff is linear in known features.

$$
u_\theta(s, a) = \phi(s, a)^\top \theta
$$

The transition kernel is the probability of the next state given the current
state and action.

$$
P_a(s, s') = \Pr(s_{t+1} = s' \mid s_t = s, a_t = a)
$$

The integrated value function solves the soft Bellman fixed point.

$$
V_\theta(s)
= \sigma \log \sum_a \exp\left(
    \frac{
        u_\theta(s, a)
        + \beta \sum_{s'} P_a(s, s') V_\theta(s')
    }{\sigma}
\right)
$$

The choice-specific value is defined as follows.

$$
Q_\theta(s, a)
= u_\theta(s, a)
  + \beta \sum_{s'} P_a(s, s') V_\theta(s')
$$

The implied conditional choice probability follows the soft-max rule.

$$
\pi_\theta(a \mid s)
=
\frac{\exp(Q_\theta(s, a) / \sigma)}
     {\sum_b \exp(Q_\theta(s, b) / \sigma)}
$$

NFXP maximizes the conditional log likelihood.

$$
\hat{\theta}
= \arg\max_\theta \sum_{i,t} \log \pi_\theta(a_{it} \mid s_{it})
$$

## Algorithm Sketch

The estimation loop first estimates transitions or accepts a supplied
transition tensor. It then initializes the reward parameters, solves the
Bellman fixed point for each candidate parameter vector, computes choice
probabilities and the likelihood, and updates the parameters until the outer
optimizer converges. The fitted object returns structural parameters, standard
errors, policy, value function, and diagnostics.

EconIRL uses a hybrid inner solver in the simulation study. It uses safe
successive approximation far from the fixed point and Newton-Kantorovich style
updates near the solution.

## Score Calculation

The score has the logit form.

$$
\psi_i(\theta)
=
\frac{1}{\sigma}
\left[
    \frac{\partial Q_\theta(s_i, a_i)}{\partial \theta}
    -
    \sum_a \pi_\theta(a \mid s_i)
    \frac{\partial Q_\theta(s_i, a)}{\partial \theta}
\right]
$$

The derivative of the value function is obtained by differentiating the fixed
point. In implementation terms, this avoids treating the inner dynamic program
as a black box.
