# Under the Hood

MPEC estimates a structural dynamic discrete choice likelihood while treating
the value function as part of the optimizer state. The estimator chooses reward
parameters and value variables jointly, subject to the Bellman fixed-point
constraint.

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

The integrated value function must satisfy the soft Bellman equation.

$$
V(s)
= \sigma \log \sum_a \exp\left(
    \frac{
        u_\theta(s, a)
        + \beta \sum_{s'} P_a(s, s') V(s')
    }{\sigma}
\right)
$$

MPEC keeps this equation as an equality constraint instead of solving it
inside each likelihood evaluation.

$$
g(\theta, V) = V - T_\theta(V) = 0
$$

The choice-specific value is defined as follows.

$$
Q_{\theta,V}(s, a)
= u_\theta(s, a)
  + \beta \sum_{s'} P_a(s, s') V(s')
$$

The implied conditional choice probability follows the soft-max rule.

$$
\pi_{\theta,V}(a \mid s)
=
\frac{\exp(Q_{\theta,V}(s, a) / \sigma)}
     {\sum_b \exp(Q_{\theta,V}(s, b) / \sigma)}
$$

MPEC solves the constrained likelihood problem.

$$
(\hat{\theta}, \hat{V})
= \arg\max_{\theta,V} \sum_{i,t}
    \log \pi_{\theta,V}(a_{it} \mid s_{it})
\quad \text{s.t.} \quad
V = T_\theta(V)
$$

## Algorithm Sketch

The estimation loop first estimates transitions or accepts a supplied
transition tensor. It initializes reward parameters and value variables, builds
the Bellman equality constraints, computes the constrained likelihood and its
derivatives, and updates the joint parameter-value vector until the optimizer
converges. The fitted object returns structural parameters, standard errors,
policy, value function, likelihood, and constraint diagnostics.

The certified validation run uses the SQP path and reports the final Bellman
constraint violation as a hard release gate. The validation claim requires both
constraint satisfaction and recovery of reward, policy, value, Q, and
counterfactual oracle objects.

## Score Calculation

For linear rewards, the score has the same logit structure as NFXP, but the
value function is part of the constrained optimizer state rather than an
implicit inner-loop solution.

The implementation computes per-observation gradients for inference after the
constrained optimum is found. Standard errors are attached to the shared
summary object so MPEC remains compatible with the same inference surface used
by NFXP and CCP.
