# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from an agent
whose behavior solves the soft Bellman equation with discount $\beta$ and
i.i.d. logit taste shocks (scale $\sigma = 1$). Known transition kernels
$P(s' \mid s, a)$ are supplied. The reward is a neural function of the state
encoding and action identity.

The agent's choice-specific value function and soft value function satisfy

$$
Q(s, a) = r_\eta(s, a) + \beta \sum_{s'} P(s' \mid s, a)\, V(s'),
\qquad
V(s) = \log \sum_a \exp\bigl(Q(s, a)\bigr),
$$

and choice probabilities are

$$
\pi(a \mid s) = \exp\bigl(Q(s, a) - V(s)\bigr).
$$

The soft value function uses log-sum-exp without an Euler-gamma constant, which
matches the package convention throughout.

## Neural Reward Parameterization

The reward function is a feedforward network $f_\eta$ with ReLU activations.
For `reward_type="state_action"` (the default), the input to the network at
each $(s, a)$ pair is the concatenation of a state feature vector $x(s)$ and
the action one-hot encoding $e(a)$:

$$
r_\eta(s, a) = f_\eta\bigl([x(s),\; e(a)]\bigr).
$$

For `reward_type="state"` the network takes only $x(s)$ and the output is
broadcast to all actions.

## Reward Normalization

A neural reward map is identified only up to an action-independent additive
function of the state (potential shaping). Pinning the reward down requires
an external normalization. The estimator imposes this by setting one action's
reward column to zero for all states:

$$
r_\eta(s, a_0) = 0 \quad \text{for all } s,
$$

where $a_0$ is the `anchor_action` argument. Alternatively, an absorbing state
row can be fixed to zero. Reward comparisons across runs or against oracle
objects are meaningful only under the same normalization.

## Occupancy Matching Objective

Let $D_{\text{data}}(s, a)$ be the empirical discounted state-action occupancy
from the demonstrations, and let $D_\pi(s, a)$ be the occupancy under the
current policy $\pi_\eta$, computed by the discounted forward pass

$$
D_\pi(s) = \rho_0(s) + \beta \sum_{s', a} D_\pi(s', a)\, P(s \mid s', a),
\qquad
D_\pi(s, a) = \pi(a \mid s)\, D_\pi(s).
$$

The MCE-IRL objective is to find $\eta$ such that $D_\pi = D_{\text{data}}$.
The training gradient with respect to the reward matrix is

$$
\nabla_R L = D_{\text{data}}(s, a) - D_\pi(s, a).
$$

## Training Loop

Because $D_\pi$ depends on the policy, which depends on the soft Bellman
solution, which depends on the reward, the gradient of a differentiable
loss through the entire solve is expensive. The implementation instead uses
a surrogate loss whose gradient matches the MCE gradient without
differentiating through the Bellman solve:

$$
L_{\text{surrogate}}(\eta) = \sum_{s, a} r_\eta(s, a)\cdot
\bigl(D_\pi(s, a) - D_{\text{data}}(s, a)\bigr).
$$

The gradient of this surrogate with respect to $\eta$ equals the chain-rule
gradient of the occupancy mismatch through the reward network. The reward
network is updated with Adam and a global gradient-norm clip.

## Pseudocode

```
for epoch in range(max_epochs):
    compute R(s,a) = f_eta([x(s), e(a)]) for all (s,a)
    solve soft Bellman: V, pi = soft_value_iteration(R, transitions)
    compute D_pi via discounted forward pass
    compute gradient: grad_R = D_data - D_pi
    compute surrogate loss: L = sum(R * grad_R)
    backprop through reward network; Adam step
    checkpoint best model by surrogate loss; early stop if no improvement
solve soft Bellman at best model to extract final policy and value
optionally project reward matrix onto linear features for theta
```

## Implementation Notes

The implementation lives in `econirl.estimators.mceirl_neural` (the
`MCEIRLNeural` class) with the lower-level MCE solver in
`econirl.estimation.mce_irl`. The soft Bellman solve uses the hybrid
value-iteration-plus-Newton-Kantorovich solver shared with NFXP and MCE-IRL.
The transitions tensor must be in `(n_actions, n_states, n_states)` orientation.
The feature projection (when `features=` is supplied) is a plain least-squares
regression of the flattened reward matrix onto the flattened feature matrix.
