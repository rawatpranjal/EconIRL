# IQ-Learn

IQ-Learn recovers an imitation policy and a Bellman-implied reward by optimizing a
soft Q-function directly against expert state-action pairs. The construction, due to
{ref}`Garg et al. (2021) <garg-2021>`, collapses the conventional adversarial min-max
between reward and policy into a single concave objective over Q alone: in a
max-entropy policy framework the optimal policy is the softmax of Q, so training the
Q-function is sufficient. The resulting Q-function encodes the imitation policy and
yields a Bellman-implied reward as a diagnostic object. It does not enforce a
structural Bellman fixed point and does not produce a point-identified reward parameter
vector.

## Source Papers

The estimator follows {ref}`Garg et al. (2021) <garg-2021>`, which introduces inverse
soft-Q learning and establishes its connection to behavioral cloning and to
f-divergence IRL objectives.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action, observed in
expert trajectories $(s, a, s')$. The discount factor is $\beta$, the logit shock
scale is $\sigma$, and the transition kernel $F_a(s' \mid s)$ gives the probability of
moving to $s'$ from $s$ under action $a$, stored in $(A, S, S)$ orientation. The soft
Q-function $Q(s, a)$ is parameterized as a free $(S \times A)$ table, a linear
combination of reward features $\varphi(s, a)^\top \theta$, or a small feedforward
network. The soft value function is $V(s) = \sigma \log \sum_a \exp(Q(s,a)/\sigma)$,
the implied policy is $\pi(a \mid s) = \exp((Q(s,a) - V(s))/\sigma)$, and the
Bellman-implied reward is $r_{\mathrm{IB}}(s, a) = Q(s, a) - \beta \sum_{s'}
F_a(s' \mid s)\, V(s')$. The divergence penalty strength is $\alpha$. The expert
occupancy measure $\rho_E(s, a) = (1 - \beta)\,\pi_E(a \mid s)\sum_{t \geq 0}
\beta^t P(s_t = s \mid \pi_E)$ is the discounted state-action visitation frequency
under the expert policy. In practice $\mathbb{E}_\rho[\cdot]$ is approximated by the
sample mean over the expert panel.

## Model

The expert data are state-action-next-state triples $(s, a, s')$ from a stationary
infinite-horizon dynamic discrete choice model. In the max-entropy policy framework
the optimal policy is the logit softmax of the true Q-function:

$$
V(s)
= \sigma \log \sum_a \exp\!\bigl(Q(s, a) / \sigma\bigr),
\qquad
\pi(a \mid s)
= \exp\!\bigl((Q(s, a) - V(s)) / \sigma\bigr).
$$

These satisfy $Q(s, a) - V(s) = \sigma \log \pi(a \mid s)$. The temporal difference
of $Q$ under the supplied transitions defines the Bellman-implied reward:

$$
r_{\mathrm{IB}}(s, a)
= Q(s, a) - \beta \sum_{s'} F_a(s' \mid s)\, V(s').
$$

If $Q$ were the true soft Bellman Q-function of the expert, $r_{\mathrm{IB}}$ would
recover the underlying reward. In practice $Q$ is fitted on expert support only and
$r_{\mathrm{IB}}$ is a diagnostic object rather than a structural estimate.

## Identification

IQ-Learn yields an imitation policy and a Bellman-implied reward under the following
conditions.

- **Max-entropy policy framework.** The expert policy is the logit softmax of the true
  Q-function; the adversarial min-max over reward-policy pairs collapses to a concave
  optimization over $Q$ alone. This is the content of Propositions 3.4 and 3.6 in Garg
  et al. (2021): the IRL saddle-point $(\pi^*, Q^*)$ is recovered by maximizing
  $\mathcal{J}^*(Q) = \mathcal{J}(\pi_Q, Q)$ over $Q$ alone, and $\mathcal{J}^*$ is
  concave in $Q$.
- **Known transitions.** The transition kernel $F_a(s' \mid s)$ is supplied externally
  in $(A, S, S)$ orientation. Rows must be stochastic; an incorrect orientation
  invalidates the Bellman-implied reward.
- **Known discount and scale.** $\beta$ and $\sigma$ are treated as fixed.
  Misspecification shifts $r_{\mathrm{IB}}$ by a multiplicative factor.
- **Expert support requirement.** The objective scores only expert $(s, a)$ pairs.
  Full state coverage ($= 1.0$) and near-full state-action coverage ($\geq 0.95$) are
  required before $r_{\mathrm{IB}}$ or counterfactual regret is interpreted;
  off-support values are extrapolated, not fitted.
- **Reward-shaping non-identification.** The objective selects a small-implied-reward
  representative on expert support but does not anchor the reward economically.
  $r_{\mathrm{IB}}$ is identified only up to behavior-preserving transformations; it
  is not a point-identified structural parameter vector.

These hold inside a finite discrete state space with stationary dynamics. The estimator
recovers the imitation policy on expert support. Structural counterfactual
interpretation requires all coverage and Bellman-object gates to pass.

## Derivation

The IRL min-max objective (Garg et al. 2021, Eq. 3) is

$$
\max_{r \in \mathcal{R}}\;\min_{\pi \in \Pi}\;
L(\pi, r)
= \mathbb{E}_{\rho_E}[r(s,a)] - \mathbb{E}_{\rho_\pi}[r(s,a)] - H(\pi) - \psi(r),
$$

where $\psi(r)$ is a convex reward regularizer and $H(\pi)$ is the causal entropy of
$\pi$. Naively solving this requires alternating between reward and policy updates.
The key insight is a three-step collapse.

**Step 1: reparameterize via the inverse soft-Bellman operator.**
Define the inverse soft-Bellman operator $\mathcal{T}^\pi : \mathbb{R}^{S \times A}
\to \mathbb{R}^{S \times A}$ by

$$
(\mathcal{T}^\pi Q)(s, a)
= Q(s, a) - \beta\,\mathbb{E}_{s' \sim P(\cdot \mid s, a)}\bigl[V^\pi(s')\bigr],
$$

where $V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[Q(s,a) - \sigma\log\pi(a \mid s)]$
is the soft value (Garg et al. 2021, before Lemma 3.2). The operator $\mathcal{T}^\pi$
inverts the soft Bellman operator $\mathcal{B}^\pi$, giving a bijection between
Q-functions and rewards (Lemma 3.2). This lets us re-parameterize the reward-policy
space $\Pi \times \mathcal{R}$ as a Q-policy space, defining the new objective
$\mathcal{J}(\pi, Q)$ (Lemma 3.3):

$$
\mathcal{J}(\pi, Q)
= \mathbb{E}_{\rho_E}\bigl[(\mathcal{T}^\pi Q)(s,a)\bigr]
  - \mathbb{E}_{\rho_\pi}\bigl[(\mathcal{T}^\pi Q)(s,a)\bigr]
  - H(\pi) - \psi(\mathcal{T}^\pi Q),
$$

with the identity $L(\pi, r) = \mathcal{J}(\pi, \mathcal{T}^{-1}r)$ for all
$r \in \mathcal{R}$ (Lemma 3.3). Simplifying using the initial state distribution
$p_0$ (Lemma A.2) yields Eq. 5 of the paper:

$$
\mathcal{J}(\pi, Q)
= \mathbb{E}_{(s,a) \sim \rho_E}\!\bigl[Q(s,a) - \beta\,\mathbb{E}_{s'}\bigl[V^\pi(s')\bigr]\bigr]
  - (1 - \beta)\,\mathbb{E}_{s_0 \sim p_0}\bigl[V^\pi(s_0)\bigr]
  - \psi(\mathcal{T}^\pi Q).
$$

**Step 2: substitute the optimal policy in closed form.**
For any fixed $Q$, $\operatorname{argmin}_{\pi \in \Pi} \mathcal{J}(\pi, Q)$ is attained
at $\pi_Q(a \mid s) = \frac{1}{Z_s}\exp(Q(s,a)/\sigma)$ with
$Z_s = \sum_b \exp(Q(s,b)/\sigma)$, the max-entropy policy for reward
$\mathcal{T}^\pi Q$ (Proposition 3.5, written here in the page's $\sigma$-scaled
convention). Substituting this policy collapses
the saddle-point to a single objective $\mathcal{J}^*(Q) = \mathcal{J}(\pi_Q, Q)$
that depends only on $Q$.

**Step 3: chi-squared substitution and the practical loss.**
For the chi-squared regularizer $\psi(r) = \alpha r^2$ we have
$\phi(x) = x - \frac{1}{4\alpha}x^2$ (Garg et al. 2021, Table 2). Substituting
$\pi_Q$ and using the offline approximation (Section 5.1 of the paper, which
replaces the initial-state term with expert samples) gives the maximization
objective (Eq. 9 $\to$ Eq. 12):

$$
\mathcal{J}^*(Q)
= \mathbb{E}_{\rho_E}\!\left[\phi\!\left(r_{\mathrm{IB}}(s,a)\right)\right]
= \mathbb{E}_{\rho_E}\!\left[r_{\mathrm{IB}}(s,a) - \frac{1}{4\alpha}\,r_{\mathrm{IB}}(s,a)^2\right],
$$

where $r_{\mathrm{IB}}(s,a) = (\mathcal{T}^{\pi_Q}Q)(s,a) = Q(s,a) - \beta
\mathbb{E}_{s'}[V^{\pi_Q}(s')]$. Rewriting as a minimization loss and using
$Q(s,a) - V^{\pi_Q}(s) = \sigma\log\pi_Q(a \mid s)$ gives Eq. 12 of the paper:

$$
\mathcal{L}(Q)
= -\mathbb{E}_{\rho_E}\!\bigl[Q(s,a) - V^{\pi_Q}(s)\bigr]
  + \frac{1}{4\alpha}\,\mathbb{E}_{\rho_E}\!\bigl[r_{\mathrm{IB}}(s,a)^2\bigr].
$$

**First-order condition.** Differentiating $\mathcal{L}(Q)$ with respect to $Q(s,a)$
and setting to zero (holding $V$ fixed at the softmax value):

$$
\frac{\partial \mathcal{L}}{\partial Q(s,a)} = 0
\;\Longrightarrow\;
-1 + \frac{1}{2\alpha}\,r_{\mathrm{IB}}(s,a) = 0
\;\Longrightarrow\;
r_{\mathrm{IB}}^*(s,a) = 2\alpha.
$$

At the optimum the implied reward on expert support equals $2\alpha$; the
divergence penalty $\alpha$ controls the implied-reward scale, not just
regularization strength.

**Note on implicit differentiation.** The implicit-differentiation step
$(I - \beta P_\pi)\,\mathrm{d}V/\mathrm{d}\theta$ arises in structural estimators
(NFXP, MPEC) where a Bellman fixed point is enforced as a hard constraint and
$\theta$ parameterizes the Bellman operator. IQ-Learn has no such constraint:
$Q$ is a free object (a table or a network), not a fixed point of any operator.
No implicit differentiation applies here; gradients flow directly through $V(s)
= \sigma\log\sum_a \exp(Q(s,a)/\sigma)$.

## Estimator

IQ-Learn minimizes over $Q$ the chi-squared objective:

$$
\mathcal{L}(Q)
= -\mathbb{E}_\rho\bigl[Q(s, a) - V(s)\bigr]
+ \frac{1}{4\alpha}\,\mathbb{E}_\rho\bigl[r_{\mathrm{IB}}(s, a)^2\bigr],
$$

where $\mathbb{E}_\rho$ averages over expert $(s, a)$ pairs. The identity
$Q(s, a) - V(s) = \sigma \log \pi(a \mid s)$ follows directly from
$V(s) = \sigma \log \sum_{a'} \exp(Q(s,a')/\sigma)$, so the first term equals
$\sigma\,\mathbb{E}_\rho[\log \pi(a \mid s)]$, the $\sigma$-scaled behavioral-cloning
log-likelihood (Garg et al. 2021, Section 5.3). The second term penalizes large
implied rewards on expert support and ensures the objective is bounded from below.
Setting $\alpha \to \infty$ removes the penalty: $\mathcal{L}(Q) \to
-\mathbb{E}_\rho[Q(s,a) - V(s)] = -\sigma\,\mathbb{E}_\rho[\log \pi(a \mid s)]$,
which is the behavioral-cloning log-likelihood (Garg et al. 2021, Section 5.3).
Standard errors are not computed; the returned standard-error array is NaN.

**Estimator note (numerical stability).** The chi-squared divergence with quadratic
temporal-difference penalty is required for bounded optimization on a free tabular
$Q$ table. The simple (TV-distance) objective,

$$
-\mathbb{E}_\rho\bigl[r_{\mathrm{IB}}(s, a)\bigr] + (1 - \beta)\,\mathbb{E}_{s_0}\bigl[V(s_0)\bigr],
$$

has no upper bound on a free tabular $Q$ and drives the optimizer to numerical
overflow; it should not be used with the tabular parameterization. This is the
negation of the paper's Eq. 9 TV objective (Garg et al. 2021), written as a
minimization; $\beta$ here corresponds to $\gamma$ in the paper.

## Algorithm

```text
Algorithm  IQ-Learn (chi-squared, tabular Q, L-BFGS-B)
Input   expert panel {(s_i, a_i)}, transitions F, discount beta, scale sigma,
        divergence penalty alpha
Output  policy pi, value V, Bellman-implied reward r_IB,
        expert state and state-action coverage fractions

1   compute expert (s, a) pairs from the panel
2   measure expert_state_coverage and expert_state_action_coverage
3   initialize Q := 0, shape (S, A)
4   minimize L(Q) using L-BFGS-B with JAX autodiff gradients:
    4a  V(s)       := sigma * logsumexp(Q(s, :) / sigma)
    4b  EV(s, a)   := sum_{s'} F_a(s' | s) V(s')
    4c  r_IB(s, a) := Q(s, a) - beta * EV(s, a)
    4d  loss       := -mean_{expert}[Q(s,a) - V(s)]
                      + (1 / (4 alpha)) * mean_{expert}[r_IB(s, a)^2]
5   after convergence:
    5a  pi(a | s) := softmax(Q(s, :) / sigma)
    5b  V(s)      := sigma * logsumexp(Q(s, :) / sigma)
    5c  r_IB(s, a) := Q(s, a) - beta * EV(s, a)
6   return pi, V, r_IB, expert_state_coverage, expert_state_action_coverage
```

The default configuration is `q_type="tabular"` with `divergence="chi2"` and
`optimizer="L-BFGS-B"`. Two alternative parameterizations are available.
`q_type="linear"` sets $Q(s, a) = \varphi(s, a)^\top \theta$ from the utility feature
matrix and optimizes with L-BFGS-B; this constrains $Q$ to the structural feature
space and propagates reward estimates to unvisited state-action pairs.
`q_type="neural"` replaces the Q table with a feedforward network and optimizes with
Adam and gradient clipping. The implementation lives in `econirl.estimation.iq_learn`.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| An imitation policy without adversarial training is required. | Structural counterfactual analysis with standard errors is required (use NFXP, CCP, or UFXP). |
| A Q-based reward diagnostic is useful alongside the imitation policy. | The Bellman fixed point must be enforced as a hard constraint (use NFXP or MPEC). |
| Transitions are available to evaluate the inverse Bellman operator. | Expert state or state-action coverage is sparse (Q and reward recovery degrade off support). |
| A Bellman-aware alternative to behavioral cloning is needed. | Structural standard errors or formal inference on reward parameters are required. |

IQ-Learn sits with AIRL, f-IRL, GLADIUS, and MCE-IRL in the IRL family. Unlike AIRL
and f-IRL it avoids adversarial training. Unlike MCE-IRL and GLADIUS it parameterizes
$Q$ directly rather than a reward function. The $\alpha \to \infty$ limit recovers
behavioral cloning, making IQ-Learn a natural bridge between pure imitation and
Bellman-aware IRL.

## Usage

```python
from econirl.estimation import IQLearnConfig, IQLearnEstimator

config = IQLearnConfig(
    q_type="tabular",
    divergence="chi2",
    alpha=1.0,
)
estimator = IQLearnEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)
print(summary.metadata["expert_state_coverage"])
print(summary.metadata["expert_state_action_coverage"])
```

The Bellman-implied reward and its projection into the utility feature basis are
available as diagnostics:

```python
r_ib = summary.metadata["raw_bellman_reward_table"]
theta_proj = summary.metadata["reward_params"]
```

These are diagnostic objects; they should not be treated as structurally recovered
parameters unless coverage and Bellman-object gates pass. To use the projected
parameters as a starting point for a structural estimator:

```python
from econirl.estimation import NFXPEstimator

nfxp_summary = NFXPEstimator().estimate(
    panel, utility, problem, transitions,
    initial_params=summary.metadata["reward_params"],
)
```

The [Quick Start](iq_learn/quick_start.md) page documents the full set of fitted
attributes and the `IQLearnEstimator` interface.

## Evidence

Behavioral recovery and counterfactual regret on the primary synthetic cell
(`canonical_low_action`, 21 states, 3 actions, 2000 individuals, 80 periods,
`q_type="tabular"`, `divergence="chi2"`, `alpha=1.0`):

| Metric | Value |
| --- | ---: |
| Policy total variation | 0.0407 |
| Type A regret (reward shift) | 0.0115 |
| Type B regret (transition change) | 0.0216 |
| Type C regret (action removal) | 0.0099 |

These checks pass their thresholds on the primary cell (policy TV threshold 0.05,
regret threshold 0.05). Reward, value, and Q recovery fail on all tested cells; the
structural gates are not satisfied. Low counterfactual regret on the primary cell
reflects that the Q-induced policy produces near-oracle welfare under the applied
interventions, not that the reward or value objects are structurally accurate.

For the cross-estimator comparison, see the
[bus engine simulation study](../simulation_studies/rust_bus.md) and the
[taxi gridworld simulation study](../simulation_studies/taxi_gridworld.md).

## References

Source papers:

- Garg, D., Chakraborty, S., Cundy, C., Song, J., and Ermon, S. (2021).
  "IQ-Learn: Inverse Soft-Q Learning for Imitation."
  _Advances in Neural Information Processing Systems_.
  {ref}`reference entry <garg-2021>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.iq_learn`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/iq_learn.py).
- Validation runner: [`validation/estimators/iq_learn/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/iq_learn/run.py).
- Results file: [`iq_learn.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/iq_learn.json).

Pages:

- [Quick Start](iq_learn/quick_start.md)
- [Pre-Estimation Checks](iq_learn/pre_estimation.md)
- [Simulation Study](iq_learn/validation.md)
- [Counterfactuals](iq_learn/counterfactuals.md)

```{toctree}
:hidden:

iq_learn/quick_start
iq_learn/pre_estimation
iq_learn/validation
iq_learn/counterfactuals
```
