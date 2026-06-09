# Under the Hood

SEES estimates a structural dynamic discrete choice likelihood while replacing
the full value vector with a sieve approximation.

## Model

The observed data are state, action, and next-state trajectories.

$$
(s_{it}, a_{it}, s_{i,t+1})
$$

The flow payoff is finite-dimensional.

$$
u_\theta(s, a) = \phi(s, a)^\top \theta
$$

The value function is approximated by basis functions.

$$
V_\alpha(s) = \Psi(s)^\top \alpha
$$

The choice-specific value is

$$
Q_{\theta,\alpha}(s,a)
= u_\theta(s,a)
  + \beta \sum_{s'} P_a(s,s') V_\alpha(s').
$$

Choice probabilities follow the soft-max rule.

$$
\pi_{\theta,\alpha}(a \mid s)
=
\frac{\exp(Q_{\theta,\alpha}(s,a) / \sigma)}
     {\sum_b \exp(Q_{\theta,\alpha}(s,b) / \sigma)}.
$$

SEES maximizes the log likelihood with a Bellman-residual penalty.

$$
(\hat{\theta}, \hat{\alpha})
= \arg\max_{\theta,\alpha}
    \ell(\theta,\alpha)
    - \omega \lVert V_\alpha - T_\theta(V_\alpha) \rVert^2.
$$

The penalty weight controls how strongly the estimated value approximation
must satisfy the Bellman equation.

## Basis Paths

For compact tabular problems, the implementation can build a basis over state
indices. For encoded-state problems, it can build an encoded-state basis from
the `DDCProblem` state encoder. The certified high-dimensional SEES validation
uses the encoded-state path with 81 basis functions and numerical rank 81.

## Inference

The lower-level estimator returns an `EstimationSummary` with reward
parameters, standard errors, policy, value function, likelihood, and metadata.
The metadata records the basis source, basis dimension, penalty weight,
Bellman violation, Bellman RMSE, and projection diagnostics.
