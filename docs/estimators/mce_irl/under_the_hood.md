# Under the Hood

MCE-IRL chooses reward parameters so the induced soft optimal policy matches
demonstrated feature expectations.

## Model

The data are state, action, and next-state trajectories $(s_{it}, a_{it},
s_{i,t+1})$. The reward is linear in known state-action features:

$$
r_\theta(s, a) = \phi(s, a)^\top \theta.
$$

Transitions $P_a(s' \mid s)$, the discount factor $\beta$, and the logit scale
$\sigma$ are fixed before optimization. For a candidate $\theta$, the
choice-specific value function solves

$$
Q_\theta(s, a) = r_\theta(s, a) + \beta \sum_{s'} P_a(s' \mid s)\, V_\theta(s'),
$$

the soft value function is

$$
V_\theta(s) = \sigma \log \sum_a \exp\!\left(\frac{Q_\theta(s, a)}{\sigma}\right),
$$

and the implied policy is the softmax of the choice-specific values:

$$
\pi_\theta(a \mid s) = \frac{\exp(Q_\theta(s, a) / \sigma)}{\sum_b \exp(Q_\theta(s, b) / \sigma)}.
$$

## Moment Condition

MCE-IRL matches discounted feature expectations. Let $D_E(s, a)$ be the
empirical discounted expert occupancy and $D_{\pi_\theta}(s, a)$ the occupancy
induced by $\pi_\theta$ and the known transition model. The estimator solves

$$
\mu_E = \mu_\theta,
\qquad
\mu_E = \sum_{s,a} D_E(s, a)\,\phi(s, a),
\quad
\mu_\theta = \sum_{s,a} D_{\pi_\theta}(s, a)\,\phi(s, a).
$$

Equivalently, the gradient of the causal-entropy dual objective is

$$
\nabla_\theta L(\theta) = \mu_E - \mu_\theta.
$$

The simulation path solves the root condition $\mu_\theta - \mu_E = 0$ directly.

## Pseudocode

```text
Input: demonstrations, reward features, transitions, discount beta, sigma
Compute demonstrated feature expectations mu_E
Initialize reward parameter vector theta
while the feature residual is not small:
    form r_theta(s, a) = phi(s, a)' theta
    solve the soft Bellman equation for Q_theta and V_theta
    compute pi_theta(a | s) = softmax(Q_theta / sigma)
    compute the occupancy measure D induced by pi_theta
    compute model feature expectations mu_theta
    update theta to reduce mu_theta - mu_E
return theta, reward table, pi_theta, V_theta, and diagnostics
```

## Implementation Notes

The implementation lives in `econirl.estimation.mce_irl`. The public simulation
path uses the root feature-matching optimizer. The occupancy measure is computed
by a forward pass under $\pi_\theta$ and the known transition tensor, which must
be in `(n_actions, n_states, n_states)` orientation.

## Identification Boundary

Action-dependent features are required for multi-action reward recovery.
State-only features broadcast across actions and leave action-specific payoff
differences unidentified. MCE-IRL rewards are interpreted relative to the
normalization encoded by the supplied feature basis and anchor.
