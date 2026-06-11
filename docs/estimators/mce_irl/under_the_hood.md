# Under the Hood

MCE-IRL chooses reward parameters so the induced soft optimal policy matches
demonstrated feature expectations.

## Optimization Setup

The observed demonstrations supply state, action, and next-state records.
Transitions, reward features, the discount factor, the logit shock scale, and
the initial state distribution are fixed before optimization.

MCE-IRL optimizes the reward parameter vector `theta`. For each candidate
`theta`, the estimator solves the soft Bellman equation, computes the induced
policy and occupancy measure, and compares model feature expectations with
demonstrated feature expectations. The public simulation path solves the root
feature-matching equation.

## Model

The observed data are state, action, and next-state trajectories.

$$
(s_{it}, a_{it}, s_{i,t+1})
$$

The reward is linear in known state-action features.

$$
r_\theta(s, a) = \phi(s, a)^\top \theta
$$

Given a candidate reward, the integrated value function solves the soft
Bellman fixed point.

$$
V_\theta(s)
= \sigma \log \sum_a \exp\left(
    \frac{
        r_\theta(s, a)
        + \beta \sum_{s'} P_a(s, s') V_\theta(s')
    }{\sigma}
\right)
$$

The implied policy is the soft-max policy over choice-specific values.

$$
\pi_\theta(a \mid s)
=
\frac{\exp(Q_\theta(s, a) / \sigma)}
     {\sum_b \exp(Q_\theta(s, b) / \sigma)}
$$

MCE-IRL matches feature counts:

$$
\mu_E
= \sum_{s,a} D_E(s, a)\phi(s, a),
\qquad
\mu_\theta
= \sum_s D_{\pi_\theta}(s)\sum_a \pi_\theta(a\mid s)\phi(s, a).
$$

The simulation path solves:

$$
\mu_\theta - \mu_E = 0.
$$

## Pseudocode

```text
Input: demonstrations, reward features, transitions, discount beta, sigma
Compute demonstrated feature expectations mu_E
Choose an initial reward parameter vector theta
while the feature residual is not small:
    form r_theta(s, a) = phi(s, a)' theta
    solve the soft Bellman equation for V_theta
    compute pi_theta(a | s) from the soft choice-specific values
    compute the occupancy measure induced by pi_theta
    compute model feature expectations mu_theta
    update theta to reduce mu_theta - mu_E
return theta, reward table, pi_theta, V_theta, and diagnostics
```

## Implementation Notes

The primer simulation uses the root feature-matching optimizer with standard
errors disabled. The public wrapper defaults are configurable, but the
simulation results file is generated from the root path above.

## Identification Boundary

Action-dependent features are required for multi-action reward recovery.
State-only features broadcast across actions can leave action-specific payoff
differences unidentified. MCE-IRL rewards are interpreted with the
normalization encoded by the supplied feature basis and anchor.
