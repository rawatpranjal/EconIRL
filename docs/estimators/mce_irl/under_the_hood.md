# Under the Hood

MCE-IRL chooses reward parameters so the induced soft optimal policy matches
demonstrated feature expectations.

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

## Algorithm Sketch

The estimator starts with demonstrated trajectories, a transition tensor, and
an explicit reward feature matrix. For each candidate parameter vector, it
computes the reward matrix, solves the soft Bellman equation, computes the
policy-induced occupancy measure, and compares model feature counts with
demonstrated feature counts.

The primer simulation uses the root feature-matching optimizer with standard
errors disabled. The public wrapper defaults are configurable, but the
known-truth artifact is generated from the root path above.

## Identification Boundary

Action-dependent features are required for multi-action reward recovery.
State-only features broadcast across actions can leave action-specific payoff
differences unidentified. MCE-IRL rewards are interpreted with the
normalization encoded by the supplied feature basis and anchor.
