# Under the Hood

## Model

The data are state-action-next-state triples $(s, a, s')$ from a stationary
agent with state-only reward $R(s)$, known transition kernels $F_a(s' \mid s)$
in shape $(A, S, S)$, discount factor $\beta$, and i.i.d. logit taste shocks
with scale $\sigma$. The agent's value function solves the soft Bellman equation

$$
V(s) = \log \sum_a \exp\!\Bigl(R(s) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr),
$$

and choice probabilities follow the softmax of choice-specific values

$$
\pi(a \mid s) \propto \exp\!\Bigl(R(s) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr).
$$

The soft Bellman uses $\log \sum \exp$ without an Euler-gamma constant, matching
the package convention throughout.

## Marginals

Define the expert state marginal $\rho_E$ as the empirical frequency of each
state in the demonstration panel. Define the model state marginal $\rho_\pi$ as
the discounted forward propagation of state visits under the soft-optimal policy
$\pi_R$ induced by the current reward:

$$
\rho_\pi(s) \propto \mu_0(s) + \sum_{t=1}^{H} \beta^t\, (P_\pi^t \mu_0)(s),
\qquad
[P_\pi]_{ss'} = \sum_a \pi(a \mid s)\, F_a(s' \mid s),
$$

where $\mu_0$ is the empirical initial-state distribution from the panel and
$H$ is the rollout horizon. The result is normalized to sum to one.

## Objective

f-IRL minimizes an f-divergence between the expert and model state marginals:

$$
\min_R\ D_f\!\bigl(\rho_E \,\|\, \rho_\pi\bigr).
$$

Five divergence families are supported, each with a closed-form gradient
direction $g(s) = \partial D_f / \partial R(s)$:

| Divergence | Gradient $g(s)$ |
| --- | --- |
| Forward KL (primary) | $\log \rho_E(s) - \log \rho_\pi(s)$ |
| Reverse KL | $\log \rho_\pi(s) - \log \rho_E(s)$ |
| Jensen-Shannon | $\log(\rho_E(s)/m(s)) - \log(\rho_\pi(s)/m(s))$, $m = (\rho_E + \rho_\pi)/2$ |
| Chi-squared | $\rho_E(s)/\rho_\pi(s) - 1$ |
| Total variation | $\operatorname{sign}(\rho_E(s) - \rho_\pi(s))$ |

## Update Rule

At each iteration the reward is updated by gradient ascent and clipped to
prevent runaway magnitudes:

$$
R^{(t+1)}(s) = \operatorname{clip}\!\Bigl(R^{(t)}(s) + \alpha\, g^{(t)}(s),\ -c,\ c\Bigr),
$$

where $\alpha$ is the learning rate and $c$ is the reward clip bound. The best
iterate over the trajectory is retained by log-likelihood (default) or
occupancy L1.

## Pseudocode

```
compute expert state marginal rho_E from the panel
initialize R(s) = 0 for all states s
for t in 1 .. max_iter:
    solve soft Bellman under R^(t) to get pi^(t) and V^(t)
    propagate pi^(t) forward to get model state marginal rho_pi^(t)
    compute gradient g^(t)(s) from the chosen f-divergence
    update R^(t+1)(s) = clip(R^(t)(s) + lr * g^(t)(s), -clip, +clip)
    record log-likelihood and occupancy L1 at this iterate
return the iterate with the best log-likelihood (or lowest occupancy L1)
```

## Implementation Notes

The implementation lives in `econirl.estimation.f_irl`. The state-marginal
forward propagation is run for `horizon` steps from the empirical initial-state
distribution and then normalized. Transitions must be provided in
`(n_actions, n_states, n_states)` orientation. The `reward_scope="state"`
option tiles the state reward vector across all actions before passing it to
the Bellman operator. The `"kl"` alias maps to `"fkl"` (forward KL) for
back-compatibility.
