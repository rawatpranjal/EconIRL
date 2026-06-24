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
The rollout horizon is $H$, the trajectory length is $T$ (the number of
time steps summed over in the gradient covariance formula, Eq. 3 of
Ni et al. 2020), the learning rate is $\alpha$, the clip bound is
$c$, and $g(s)$ is the per-state reward update direction, equal to the
instantiation of $h_f\!\bigl(\rho_E(s)/\rho_\pi(s)\bigr)$ from Theorem 4.1
of Ni et al. (2020) in the tabular case (see Estimator section).

## Model

The observed data are state-action trajectories from a stationary agent with
state-only reward $R(s)$, known transition kernels $F_a(s' \mid s)$, discount
factor $\beta$, and i.i.d. logit taste shocks. The agent's value function
solves the soft Bellman fixed point:

$$
V(s) = \log \sum_a \exp\!\Bigl(R(s) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr).
$$

The log-sum-exp form uses unit logit scale, matching the package convention
throughout. The package fixes the entropy temperature $\alpha = 1$
throughout; the general $\alpha$ form appears in Ni et al. (2020) Section 3
as the MaxEnt RL objective $V(s) = \tfrac{1}{\alpha}\log\sum_a \exp\!\bigl(\alpha(R(s) + \beta\sum_{s'} F_a(s'\mid s)\,V(s'))\bigr)$.

Choice probabilities follow the softmax of choice-specific values:

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
+ \sum_{t=1}^{H} \beta^t\,(\mu_0^\top P_\pi^t)(s),
\qquad
[P_\pi]_{ss'} = \sum_a \pi(a \mid s)\, F_a(s' \mid s),
$$

normalized to sum to one over $s$. Here $\mu_0$ is treated as a row vector
and $\mu_0^\top P_\pi^t$ denotes right-multiplication, matching the
implementation's `mu @ P_pi` convention. The canonical instance is the paper-faithful
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

**Analytic gradient (Theorem 4.1, Ni et al. 2020).** The analytic gradient of
the f-divergence objective $L_f(\theta) = D_f(\rho_E \| \rho_\theta)$ with
respect to the reward parameters $\theta$ is (Ni et al. 2020, Eq. 3):

$$
\nabla_\theta L_f(\theta)
= \frac{1}{\alpha T}
\operatorname{cov}_{\tau \sim \rho_\theta(\tau)}\!\!\left(
  \sum_{t=1}^T h_f\!\!\left(\frac{\rho_E(s_t)}{\rho_\theta(s_t)}\right),\;
  \sum_{t=1}^T \nabla_\theta r_\theta(s_t)
\right),
$$

where $h_f(u) \triangleq f(u) - f'(u)\,u$ is the generator-derived function
for each f-divergence family (Ni et al. 2020, Table 2; proof in Appendix A).
In the tabular case with a fixed state-only reward the covariance simplifies:
the per-state reward update direction $g(s) = h_f(\rho_E(s)/\rho_\pi(s))$
is the instantiation of $h_f$ at the density ratio for that state.

Two instantiations from the paper:

- **Forward KL**: $f(u) = u\log u$, so $f'(u) = 1 + \log u$ and
  $h_\text{FKL}(u) = u\log u - (1+\log u)u = -u$.
  Then $g(s) = h_\text{FKL}(\rho_E/\rho_\pi) = -(\rho_E/\rho_\pi)$.
  Under gradient *descent* on $D_f$ the update is
  $R(s) \leftarrow R(s) - \alpha \cdot (-\rho_E(s)/\rho_\pi(s))$,
  i.e., reward increases where $\rho_E(s) > \rho_\pi(s)$.
  In the tabular case the reward at each state $s$ is a scalar
  parameter, so $\nabla_\theta r_\theta(s_t) = \mathbf{e}_{s_t}$ (the
  unit basis vector for state $s_t$). Substituting into the covariance
  formula and switching to gradient *ascent* gives the update direction
  $+\rho_E(s)/\rho_\pi(s)$ per state. Because the covariance is linear
  in the first argument and $\rho_E,\rho_\pi > 0$, taking the log of
  the ratio yields an equivalent monotone direction:
  $g(s) = \log\rho_E(s) - \log\rho_\pi(s)$.
  This log form is numerically stabler (avoids large ratio spikes) and
  is the update used in the package implementation.

- **Reverse KL**: $f(u) = -\log u$, so $f'(u) = -1/u$ and
  $h_\text{RKL}(u) = -\log u - (-1/u)u = 1 - \log u$.
  The constant 1 drops under the covariance (it does not covary with
  $\nabla_\theta r_\theta$), giving update direction
  $g(s) = -\log(\rho_E(s)/\rho_\pi(s)) = \log\rho_\pi(s) - \log\rho_E(s)$.

Five divergence families are supported. The table shows the generator $f(u)$,
the derived $h_f(u) \triangleq f(u) - f'(u)\,u$, and the resulting
per-state gradient direction $g(s)$ used in the package. The FKL, RKL, and JS
entries are from Ni et al. (2020) Table 2; the chi-squared and total variation
entries are standard f-divergence generators derived by the same formula:

| Divergence | $f(u)$ | $h_f(u)$ | Gradient $g(s)$ |
| --- | --- | --- | --- |
| Forward KL (default, `"fkl"`) | $u\log u$ | $-u$ | $\log \rho_E(s) - \log \rho_\pi(s)$ |
| Reverse KL (`"rkl"`) | $-\log u$ | $1 - \log u$ | $\log \rho_\pi(s) - \log \rho_E(s)$ |
| Jensen-Shannon (`"js"`) | $u\log u - (1+u)\log\tfrac{1+u}{2}$ | $-\log(1+u)$ | $-\log\!\bigl(1 + \rho_E(s)/\rho_\pi(s)\bigr)$ |
| Chi-squared (`"chi2"`) | $(u-1)^2$ | $1 - u^2$ | $\rho_E(s)/\rho_\pi(s) - 1$ |
| Total variation (`"tv"`) | $\tfrac{1}{2}\lvert u-1\rvert$ | $\tfrac{1}{2}\operatorname{sign}(1-u)$ | $\operatorname{sign}(\rho_E(s) - \rho_\pi(s))$ |

**Note on Jensen-Shannon.** The JS gradient $g(s) = -\log(1 + \rho_E/\rho_\pi)$ from
$h_\text{JS}(u) = -\log(1+u)$ (Ni et al. 2020, Table 2) is *not* the same as
the FKL gradient $\log(\rho_E/\rho_\pi)$. The current package implementation
computes $\log(\rho_E/m) - \log(\rho_\pi/m)$ with $m = (\rho_E + \rho_\pi)/2$,
which simplifies algebraically to $\log(\rho_E/\rho_\pi)$, the same direction as FKL.
This is a known implementation divergence from the paper's Table 2 formula; the
JS and FKL variants produce identical update directions in the current code.
The JS and FKL divergences differ in their Hessian (curvature at optimum) but
not in their gradient direction under this implementation.

**Density Ratio Estimation (sample regime).** When expert state samples
$s_E$ are provided rather than the analytic density $\rho_E(s)$, the paper
fits a discriminator $D_\omega(s)$ in each iteration by maximizing the binary
cross-entropy (Ni et al. 2020, Section 4.2, Eq. 4):

$$
\max_{D_\omega}\;
\mathbb{E}_{s \sim s_E}\!\bigl[\log D_\omega(s)\bigr]
+ \mathbb{E}_{s \sim \rho_\theta}\!\bigl[\log(1 - D_\omega(s))\bigr].
$$

The optimal discriminator satisfies $D^*_\omega(s) = \rho_E(s)/(\rho_E(s) +
\rho_\theta(s))$ (the standard logit solution for binary density-ratio
estimation), so the density ratio entering $h_f$ is recovered as
$\rho_E(s)/\rho_\theta(s) \approx D_\omega(s)/(1 - D_\omega(s))$.
The package's current implementation uses the empirical-frequency path
(analytic $\rho_E$ from the panel) and does not implement the discriminator
path.

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
        [package adaptation: tabular value iteration replaces the paper's MaxEnt RL / SAC
         inner loop from Algorithm 1 of Ni et al. (2020); the substitution is exact in the
         finite discrete case]
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
