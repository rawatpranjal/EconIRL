# f-IRL

f-IRL learns a tabular reward function by minimizing an f-divergence between
the expert state marginal and the state marginal induced by the current reward
and transition model. The method belongs to the behavioral IRL family: it
recovers a reward that reproduces the expert's distributional behavior rather
than structural utility parameters. The primary validated scope is
state-marginal matching with a state-only reward; action-dependent structural
DDC recovery remains a diagnostic exercise under current evidence.

## Source Papers

The estimator follows {ref}`Ni et al. (2020) <ni-2020>`, which introduces
f-IRL and the state-marginal matching approach via f-divergence minimization
for imitation learning.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action. The
reward function is $R(s)$ for the state-only scope or $R(s, a)$ for the
state-action scope. The transition kernel $F_a(s' \mid s)$ gives the
probability of moving to $s'$ from $s$ under action $a$, stored in $(A, S, S)$
orientation. The discount factor is $\beta$. The integrated value function is
$V(s)$ and the soft-optimal policy is $\pi(a \mid s)$. The expert state
marginal is $\rho_E(s)$, the model state marginal under the current reward is
$\rho_\pi(s)$, and the initial-state distribution from the panel is $\mu_0$.
The rollout horizon is $H$, the learning rate is $\alpha$, the clip bound is
$c$, and the f-divergence gradient direction with respect to the reward is
$g(s)$.

## Model

The observed data are state-action trajectories from a stationary agent with
state-only reward $R(s)$, known transition kernels $F_a(s' \mid s)$, discount
factor $\beta$, and i.i.d. logit taste shocks. The agent's value function
solves the soft Bellman fixed point:

$$
V(s) = \log \sum_a \exp\!\Bigl(R(s) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr).
$$

The log-sum-exp form uses unit logit scale, matching the package convention
throughout. Choice probabilities follow the softmax of choice-specific values:

$$
\pi(a \mid s) \propto \exp\!\Bigl(R(s) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr).
$$

The expert state marginal $\rho_E$ is the empirical frequency of each state in
the demonstration panel. The model state marginal $\rho_\pi$ is the discounted
forward propagation of state visits under the soft-optimal policy:

$$
\rho_\pi(s)
\;\propto\;
\mu_0(s)
+ \sum_{t=1}^{H} \beta^t\,(P_\pi^t \mu_0)(s),
\qquad
[P_\pi]_{ss'} = \sum_a \pi(a \mid s)\, F_a(s' \mid s),
$$

normalized to sum to one over $s$. The canonical instance is the paper-faithful
synthetic cell: eight states, three actions, a state-only reward, and
deterministic transitions that fully specify the data-generating process.

## Identification

f-IRL identifies a behavioral reward under the following assumptions.

- **Stationarity.** The data-generating process is a stationary Markov decision
  process and the expert follows a stationary policy.
- **Additive separability.** The per-period payoff includes additive i.i.d.
  logit taste shocks, inducing a soft-optimal policy consistent with the soft
  Bellman equation above.
- **Exogenous transitions.** The transition kernel $F_a(s' \mid s)$ is supplied
  externally or estimated in a prior stage, independent of the reward.
- **State coverage.** Each state in the expert policy's support must appear in
  the demonstration panel. States absent from the panel carry zero expert
  marginal mass; forward KL assigns unbounded cost to the model visiting them.
- **Behavioral identification.** The recovered reward is identified only up to
  behavior-preserving transformations. It is not point-identified as a
  structural utility parameter.
- **Marginal scope.** The primary validated case matches state marginals with a
  state-only reward. The action-dependent diagnostic cell fails the reward-range
  check with a reward range of 0.000 and is not structural evidence.

These hold inside a finite discrete state space with a known fixed discount
factor $\beta$. Given them, the recovered policy reproduces the expert's state
distribution. Identification weakens under thin state coverage or when
unvisited states appear in the support of the transition kernel.

## Estimator

f-IRL minimizes an f-divergence between the expert and model state marginals:

$$
\min_R\; D_f\!\bigl(\rho_E \;\|\; \rho_\pi(R)\bigr).
$$

Five divergence families are supported. Each has a closed-form gradient
direction $g(s) = \partial D_f / \partial R(s)$:

| Divergence | Gradient $g(s)$ |
| --- | --- |
| Forward KL (default, `"fkl"`) | $\log \rho_E(s) - \log \rho_\pi(s)$ |
| Reverse KL (`"rkl"`) | $\log \rho_\pi(s) - \log \rho_E(s)$ |
| Jensen-Shannon (`"js"`) | $\log(\rho_E(s)/m(s)) - \log(\rho_\pi(s)/m(s))$, $m = (\rho_E + \rho_\pi)/2$ |
| Chi-squared (`"chi2"`) | $\rho_E(s)/\rho_\pi(s) - 1$ |
| Total variation (`"tv"`) | $\operatorname{sign}(\rho_E(s) - \rho_\pi(s))$ |

The reward is updated by gradient ascent with a clip bound $c$:

$$
R^{(t+1)}(s)
= \operatorname{clip}\!\Bigl(R^{(t)}(s) + \alpha\, g^{(t)}(s),\; -c,\; c\Bigr).
$$

The best iterate over the full trajectory is retained by log-likelihood
(default) or by occupancy L1, depending on the `selection_metric` parameter.

## Algorithm

```text
Algorithm  f-IRL (forward KL state-marginal matching, primary validated variant;
           requires marginal_space='state', reward_scope='state';
           FIRLEstimator() defaults to marginal_space='state_action')
Input   panel {(s_it, a_it)}, transitions F in (A, S, S) orientation,
        discount beta, learning rate alpha, clip bound c,
        rollout horizon H, maximum iterations T
Output  R_star (tabular reward), pi_star (policy), V_star (value)

1   compute rho_E(s) from the empirical state frequencies in the panel
2   initialize R^(0)(s) = 0 for all s;  best_score := -inf
3   for t = 1 .. T do
4       tile R^(t-1) across actions to get reward matrix (if reward_scope="state")
5       solve soft Bellman under R^(t-1) via value iteration to get V^(t) and pi^(t)
6       compute P_pi^(t)(s,s') := sum_a pi^(t)(a|s) * F_a(s,s')
7       propagate rho_pi^(t) via H steps from mu_0 under P_pi^(t), then normalize
8       compute g^(t)(s) := log rho_E(s) - log rho_pi^(t)(s)     [forward KL]
9       evaluate log-likelihood LL^(t) on the panel under pi^(t)
10      if LL^(t) > best_score:
11          best_score := LL^(t);  R_star := R^(t-1);  pi_star := pi^(t);  V_star := V^(t)
12      update R^(t)(s) := clip(R^(t-1)(s) + alpha * g^(t)(s), -c, +c)
13  return R_star, pi_star, V_star
```

The default divergence is `f_divergence="fkl"` (forward KL). The legacy alias
`"kl"` resolves to `"fkl"` for back-compatibility. Four additional variants are
available: `"rkl"` (reverse KL, mode-seeking), `"js"` (Jensen-Shannon,
symmetric), `"chi2"` (chi-squared), and `"tv"` (total variation). All share
the same loop structure; only the gradient expression in step 8 changes.
Switching `selection_metric` to `"occupancy_l1"` replaces log-likelihood as the
best-iterate criterion with the smallest marginal L1 distance. When
`reward_scope="state"` the reward vector $R(s)$ is tiled across all actions
before being passed to the Bellman operator (step 4). The implementation lives
in `econirl.estimation.f_irl`.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| The reward target is state-only. | Action-dependent structural DDC reward recovery is the goal. |
| State-marginal matching is the study question. | Feature-expectation matching is preferred. |
| Transitions are known or pre-estimated. | Expert data is too sparse to estimate a reliable state marginal. |
| Multiple f-divergence choices are required. | Standard errors on recovered parameters are required. |
| A divergence-controlled imitation baseline is needed. | Counterfactual re-solving in a structural model is the primary goal. |

f-IRL sits in the behavioral IRL family alongside MCE-IRL and GLADIUS. It
differs from MCE-IRL and MaxEnt-IRL in objective: those methods match feature
expectations, while f-IRL matches state marginals directly and requires no
feature specification. Against GLADIUS and AIRL, the difference is
architecture: f-IRL is tabular and gradient-based, with no discriminator
network. It is simpler to configure and inspect, at the cost of the scalability
those neural methods offer. Action-dependent structural DDC reward recovery is
outside the validated scope.

## Usage

```python
from econirl.estimation import FIRLEstimator

estimator = FIRLEstimator(
    f_divergence="fkl",      # forward KL (primary validated divergence)
    marginal_space="state",  # match state marginals
    reward_scope="state",    # learn a state-only reward
    lr=0.5,
    max_iter=250,
)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)                        # choice probabilities (S, A)
print(summary.metadata["occupancy_l1"])      # state marginal L1 distance
print(summary.metadata["reward_range"])      # reward range; near-zero indicates failure
print(summary.metadata["reward_matrix"])     # tabular reward (S, A)
```

Counterfactual evaluation re-solves the fitted policy under modified
transitions and reads the new policy:

```python
# Type B counterfactual: changed transition model
summary_cf = FIRLEstimator(
    f_divergence="fkl",
    marginal_space="state",
    reward_scope="state",
    lr=0.5,
    max_iter=250,
).estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions_modified,
)
print(summary_cf.policy)    # policy induced by the modified transitions
```

The [Quick Start](f_irl/quick_start.md) page documents the full set of fitted
attributes and divergence options.

## Evidence

Behavioral recovery is measured on the paper-faithful synthetic cell
(`f_irl_paper_state_marginal`): eight states, three actions, a state-only
reward, deterministic transitions, and fully specified oracle objects for
policy, value, and three counterfactual families. All results use the forward
KL divergence with `marginal_space="state"` and `reward_scope="state"`.

| Metric | Value |
| --- | ---: |
| State marginal L1 | 0.000260 |
| Policy total variation | 0.0121 |

Counterfactual behavior against exact oracle objects:

| Counterfactual | Policy TV | Regret | Value RMSE |
| --- | ---: | ---: | ---: |
| Type A (reward shift) | 0.0102 | 0.00708 | 0.00710 |
| Type B (transition change) | 0.0151 | 0.01236 | 0.01237 |
| Type C (action removal) | 0.00770 | 0.00273 | 0.00277 |

Low regret across all three intervention types indicates that the soft-optimal
policy recovered under the state-marginal objective behaves well in each
intervened world. The action-dependent DDC diagnostic cell
(`canonical_low_action`) fails the reward-range check and is not included in
this evidence. For the cross-estimator comparison, see the
[taxi gridworld simulation study](../simulation_studies/taxi_gridworld.md).

## References

Source papers:

- Ni, T., Sikchi, H., Wang, Y., Gupta, T., Lee, L., and Eysenbach, B. (2020).
  f-IRL: Inverse Reinforcement Learning via State Marginal Matching.
  _Proceedings of the 4th Conference on Robot Learning_.
  {ref}`reference entry <ni-2020>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.f_irl`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/f_irl.py).
- Validation runner: [`validation/estimators/f_irl/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/f_irl/run.py).
- Results file: [`f_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/f_irl.json).

Pages:

- [Quick Start](f_irl/quick_start.md)
- [Pre-Estimation Checks](f_irl/pre_estimation.md)
- [Simulation Study](f_irl/validation.md)
- [Counterfactuals](f_irl/counterfactuals.md)

```{toctree}
:hidden:

f_irl/quick_start
f_irl/pre_estimation
f_irl/validation
f_irl/counterfactuals
```
