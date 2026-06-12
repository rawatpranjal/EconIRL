# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from a stationary
infinite-horizon dynamic discrete choice model. A discount factor $\beta$,
transition kernels $F_a(s' \mid s)$ with orientation $(A, S, S)$, and i.i.d.
logit taste shocks with scale $\sigma$ are assumed known. Instead of
parameterizing a reward function directly, IQ-Learn parameterizes the soft
Q-function $Q(s, a)$ and derives the policy and implied reward from it.

## Soft Value and Policy

Given $Q$, the soft value function and the implied policy are:

$$
V(s) = \sigma \log \sum_a \exp\!\bigl(Q(s, a) / \sigma\bigr),
\qquad
\pi(a \mid s) = \exp\!\bigl((Q(s, a) - V(s)) / \sigma\bigr).
$$

These satisfy $Q(s, a) - V(s) = \sigma \log \pi(a \mid s)$.

## Inverse Bellman Reward

The temporal difference of $Q$ under the transitions defines the reward
implied by $Q$ through the inverse Bellman operator:

$$
r_{\mathrm{IB}}(s, a)
= Q(s, a) - \beta \sum_{s'} F_a(s' \mid s)\, V(s').
$$

If $Q$ is a true soft Bellman Q-function, $r_{\mathrm{IB}}$ recovers the
underlying reward. In practice it is a diagnostic: it is fitted on expert
support and can be unreliable off support.

## Chi-Squared Objective

IQ-Learn minimizes over $Q$:

$$
\mathcal{L}(Q)
= -\mathbb{E}_\rho\bigl[Q(s, a) - V(s)\bigr]
+ \frac{1}{4\alpha}\,\mathbb{E}_\rho\bigl[r_{\mathrm{IB}}(s, a)^2\bigr],
$$

where $\mathbb{E}_\rho$ averages over expert $(s, a)$ pairs. Since
$Q(s, a) - V(s) = \sigma \log \pi(a \mid s)$, the first term is behavioral
cloning scaled by $\sigma$. The second term penalizes large implied rewards on
expert support, which bounds the objective and prevents $Q$ from drifting to
numerical overflow on a free tabular table.

The simple (TV-distance) objective replaces the quadratic penalty with
$-\mathbb{E}_\rho[r_{\mathrm{IB}}(s, a)] + (1 - \beta)\,\mathbb{E}_{s_0}[V(s_0)]$,
which has no upper bound on a free tabular $Q$ and should not be used there.

## Q Parameterizations

Three parameterizations are supported:

- **Tabular**: $Q$ is a free $(S \times A)$ matrix. No structure is imposed;
  the recovered reward is per-cell and does not propagate to unvisited
  state-action pairs. Optimized by L-BFGS-B.
- **Linear**: $Q(s, a) = \varphi(s, a)^\top \theta$ where $\varphi$ is the
  feature matrix from the utility specification. This constrains $Q$ to the
  structural feature space and propagates through features to unvisited pairs.
  Optimized by L-BFGS-B.
- **Neural**: $Q(s, \cdot) = f_\psi(s)$ for a small feedforward network.
  Optimized by Adam.

## Pseudocode

```
compute expert (s, a) pairs from the panel
compute expert_state_coverage and expert_state_action_coverage
initialize Q (zeros for tabular/linear, random for neural)
minimize L(Q) over the chosen parameterization
extract policy: pi(a|s) = softmax(Q(s,:) / sigma)
extract value: V(s) = sigma * logsumexp(Q(s,:) / sigma)
extract implied reward: r_IB(s,a) = Q(s,a) - beta * sum_{s'} F_a(s'|s) V(s')
report policy, value, r_IB, and coverage fractions
```

## Implementation Notes

The implementation lives in `econirl.estimation.iq_learn`. Coverage fractions
are computed before optimization and exposed through
`summary.metadata["expert_state_coverage"]` and
`summary.metadata["expert_state_action_coverage"]`. Standard errors are not
computed; the returned standard error array is NaN. The raw Bellman implied
reward and its projection into the utility feature basis are both stored in
`summary.metadata` for diagnostic use.
