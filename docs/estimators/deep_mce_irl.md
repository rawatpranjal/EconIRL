# Neural MCE-IRL

## Important Links

- [Quick Start](deep_mce_irl/quick_start.md)
- [Pre-Estimation Checks](deep_mce_irl/pre_estimation.md)
- [Simulation Study](deep_mce_irl/validation.md)
- [Wulfmeier-Shaped Study](deep_mce_irl/wulfmeier_objectworld.md)
- [Counterfactuals](deep_mce_irl/counterfactuals.md)

Neural MCE-IRL (also called Deep MCE-IRL) recovers a neural reward map from
observed choices by matching
the discounted state-action occupancy of an entropy-regularized policy to the
occupancy observed in demonstrations. It uses the same soft Bellman planning
and occupancy-matching objective as MCE-IRL, but the reward is a feedforward
network rather than a dot product with fixed features. The validated object is
the anchored reward matrix and the behavior it induces; raw network weights are
not a structural estimand.

For a runnable fit with exact output, start with the
[Quick Start](deep_mce_irl/quick_start.md).

## Source Papers

The estimator draws on {ref}`Ziebart (2010) <ziebart-2010>`, which establishes
the maximum causal entropy IRL framework, the soft Bellman planning operator,
and the occupancy-matching gradient. {ref}`Wulfmeier, Ondruska, and Posner
(2015) <wulfmeier-2015>` introduced the neural reward parameterization into
this framework, noting that the occupancy-mismatch gradient flows through the
reward matrix entry by entry and can be backpropagated through any
differentiable reward network.

## Theory Connections

For the proof route behind this page, start with
[Soft Bellman and DDC-MaxEnt Equivalence](../theory/soft_bellman_equivalence.md)
for the soft planning identity,
[Identification and Anchors](../theory/identification.md) for why the reward
matrix must be anchored, and
[IRL Identification Boundaries](../theory/irl_boundaries.md) for why neural
weights are not themselves the identified structural object. Use
[Reward Projection and Feature Rank](../theory/reward_projection.md) for the
distinction between a neural reward map and a finite projected parameter.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action,
observed for individual $i$ in period $t$. The state encoder $x(s)$ maps
state indices to feature vectors and $e(a)$ denotes the action one-hot
encoding. The neural reward map $r_\eta(s, a)$ is parameterized by network
weights $\eta$. The discount factor is $\beta$ and the logit shock scale is
$\sigma = 1$ throughout. The transition kernel $P(s' \mid s, a)$ gives the
probability of moving to $s'$ from $s$ under action $a$, supplied in
$(A, S, S)$ orientation. The soft value function is $V(s)$, the
choice-specific value is $Q(s, a)$, and the conditional choice probability is
$\pi(a \mid s)$. The empirical discounted state-action occupancy from
demonstrations is $D_\text{data}(s, a)$ and the model occupancy under the
current policy is $D_\pi(s, a)$. The initial-state distribution is
$\rho_0(s)$, giving the probability of starting in state $s$.

## Model

The observed data are state, action, and next-state trajectories
$(s_{it}, a_{it}, s_{i,t+1})$. The agent's behavior solves a soft Bellman
problem with discount $\beta$ and i.i.d. logit taste shocks with scale
$\sigma = 1$. The reward is a feedforward network $f_\eta$ with ReLU
activations. For `reward_type="state_action"` (the default), the input at
each $(s, a)$ pair is the concatenation of the state features and the action
one-hot:

$$
r_\eta(s, a) = f_\eta\bigl([x(s),\; e(a)]\bigr).
$$

The choice-specific value and soft value function satisfy

$$
Q(s, a) = r_\eta(s, a) + \beta \sum_{s'} P(s' \mid s, a)\, V(s'),
\qquad
V(s) = \log \sum_a \exp\bigl(Q(s, a)\bigr).
$$

The log-sum-exp form of $V$ follows from entropy-regularized planning
(Ziebart 2010, ch. 5): the agent maximizes expected reward minus the KL
divergence to a uniform policy, and the resulting soft Bellman backup has
log-sum-exp as its fixed-point operator. The conditional choice probability
follows the logit form:

$$
\pi(a \mid s) = \exp\bigl(Q(s, a) - V(s)\bigr).
$$

For `reward_type="state"`, the network takes only $x(s)$ and the output is
broadcast to all actions, restricting the reward to be action-independent.
The soft value function uses log-sum-exp without an Euler-gamma constant,
following the package convention throughout.

## Identification

This is the section that says when an anchored neural reward map is interpretable,
and when it is only a behavior-fitting object.

Deep MCE-IRL identifies the anchored reward map $r_\eta$ under the following
conditions.

- **Known transitions.** The transition kernel $P(s' \mid s, a)$ is supplied
  externally and is not estimated jointly with the reward.
- **Causal separability.** Per-period payoffs are the systematic reward plus
  an additive i.i.d. Type-I extreme-value shock with scale $\sigma = 1$,
  drawn independently across choices. The agent acts before observing future
  randomness, so the soft Bellman operator respects the causal direction of
  time in the decision problem.
- **Reward normalization.** Behavior alone cannot distinguish rewards related
  by potential shaping:

  $$
  \widetilde r(s,a)
  = r(s,a) + \Phi(s)
  - \beta\mathbb{E}[\Phi(s')\mid s,a].
  $$

  These rewards induce the same policy under the observed transition law. For
  state-action rewards, the package imposes the DDC normalization

  $$r_\eta(s, a_0) = 0 \quad \text{for all } s.$$

  With known transitions, a known anchor payoff, and expert support on the
  anchor action, this condition selects an identified reward representative on
  the covered support. Reward-map comparisons require the same transition law
  and normalization. For state-only rewards, the package subtracts the reward
  at `anchor_state` from every state. This removes a global additive constant
  without changing the induced policy.
- **State encoding supplied.** The state encoder $x(s)$ is supplied
  externally. The reward map is identified relative to the chosen encoding;
  a different encoding produces a reward map on a different domain.
- **Neural weight non-identification.** Multiple network parameter vectors
  $\eta$ can represent the same reward matrix. The identified object is the
  reward matrix under the chosen normalization, not the raw network weights.

These conditions hold inside a finite discrete state space with a stationary
environment and a fixed discount factor $\beta$. At the population level,
these conditions identify the anchored reward matrix and its induced policy,
value, and Q functions on the covered support.
Identification weakens under thin state-action coverage, an inconsistent
normalization anchor, or a poorly conditioned state encoding. A finite
projection can summarize one fitted reward surface when its design is
well-conditioned. It is descriptive, not a structural parameter estimate.
See the [Pre-Estimation Checks](deep_mce_irl/pre_estimation.md) page.

## Estimator

The MCE-IRL objective matches the expert and model state-action occupancies.
Under the maximum causal entropy model, the probability of a trajectory is
proportional to the exponentiated sum of rewards along it
(Ziebart 2008; Wulfmeier 2015, eq. 1), so the log-likelihood of
demonstrations $\mathcal{D}$ is

$$
L(\eta) = \sum_{i,t} r_\eta(s_{it}, a_{it}) - \log Z(\eta),
$$

where $Z(\eta)$ is the partition function over all trajectories
(Ziebart 2010, ch. 5; Wulfmeier 2015, eq. 8). Maximizing $L(\eta)$ with
respect to the network weights $\eta$ is the MCE-IRL objective. The surrogate
below is used in place of differentiating through $Z$ directly, which would
require backpropagating through the soft Bellman solve.

The implementation normalizes empirical discounted occupancy to sum to one.
Let

$$
C_\mathcal{D} = \sum_{i,t}\beta^t.
$$

Then

$$
D_\text{data}(s, a)
= \frac{1}{C_\mathcal{D}}
  \sum_{i,t} \beta^t \,\mathbf{1}[s_{it} = s,\; a_{it} = a].
$$

For the model, first compute the unnormalized discounted state occupancy

$$
\widetilde D_\pi(s)
= \rho_0(s) + \beta \sum_{s',a}
  \widetilde D_\pi(s',a)P(s\mid s',a),
\qquad
\widetilde D_\pi(s,a)=\pi(a\mid s)\widetilde D_\pi(s).
$$

Then normalize:

$$
D_\pi(s)
= \frac{\widetilde D_\pi(s)}
       {\sum_u\widetilde D_\pi(u)},
\qquad
D_\pi(s,a)=\pi(a\mid s)D_\pi(s).
$$

At each epoch, the estimator forms the normalized occupancy mismatch

$$
\Delta_D(s,a)=D_\pi(s,a)-D_\text{data}(s,a).
$$

It treats this quantity as a fixed reward sensitivity and backpropagates the
surrogate

$$
L_\text{surrogate}(\eta)
= \sum_{s,a} r_\eta(s,a)\Delta_D(s,a)
$$

through the reward network. This is the occupancy-matching update used by the
estimator. The implementation does not evaluate the finite-trajectory
partition function directly.

## Algorithm

```text
Algorithm  Deep MCE-IRL (neural maximum causal entropy IRL)
Input   panel {(s_it, a_it)}, state encoder x, transitions P,
        discount beta, anchor action a_0, architecture (H hidden, L layers)
Output  reward matrix R_hat(s,a), policy pi, value V

1   initialize reward network f_eta with H hidden units and L layers
2   compute D_data(s,a) from the demonstration panel (discounted occupancy)
3   compute rho_0 from the initial states in the panel
4   initialize AdamW optimizer with global gradient-norm clip
5   set best_moment_loss = infinity,  patience_counter = 0
6   for epoch = 1, ..., max_epochs                 # outer loop: AdamW descent
7       R(s,a) := f_eta([x(s), e(a)])  for all (s,a)   # neural reward matrix
8       set R(s, a_0) := 0             # enforce anchor normalization
9       solve V, pi via hybrid soft value iteration (R, P)     # inner Bellman
10      compute D_pi(s,a) via discounted forward pass using pi and P
11      grad_R(s,a) := D_pi(s,a) - D_data(s,a)         # occupancy mismatch
12      surrogate := sum_{s,a} R(s,a) * grad_R(s,a)
13      backpropagate surrogate;  mask gradients for R(s,a_0) to zero;  AdamW step
14      moment_loss := sum_{s,a} grad_R(s,a)^2
15      if moment_loss < best_moment_loss - tol:  update checkpoint;  patience_counter := 0
16      else:  patience_counter := patience_counter + 1
17      if patience_counter >= patience:  break           # early stopping
18  restore best checkpoint
19  re-solve V, pi at best R via hybrid soft value iteration
20  return R_hat := R(s,a), pi, V
```

Gradients with respect to entries $R(s, a_0)$ are masked to zero before the
AdamW step (step 13), so the anchor normalization is enforced throughout
training, not only at inference.

The inner solve in steps 9 and 19 defaults to `inner_solver="hybrid"`:
successive approximation while the Bellman residual is above a switch
tolerance, then Newton-Kantorovich steps near the fixed point. The alternative
`inner_solver="value"` uses plain value iteration throughout, which is
robust from any start but converges more slowly near the solution. The outer
optimizer is AdamW with a global gradient-norm clip of 1.0, implemented via
Equinox and Optax.

## System View

Neural MCE-IRL keeps the MCE-IRL training logic but replaces the linear reward
basis with a neural reward map. The policy is still produced by a soft dynamic
program, so the transition model remains part of the estimator.

```text
Expert demonstrations
Known transition model, state/action encodings, discount factor
        |
        v
Neural network proposes a reward map
        |
        v
Solve the soft dynamic program under that map
        |
        v
Compare model occupancy to expert occupancy
        |
        v
Backpropagate the occupancy mismatch into the reward network
        |
        v
Anchored reward matrix and induced policy
```

The fitted object is the anchored reward matrix on the state-action grid. Many
network weights can represent the same matrix, so the weights themselves are not
the thing to interpret.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| Transitions are known or supplied. | Transitions must be estimated jointly. |
| The reward is nonlinear in the available state encodings. | A linear reward table is adequate (use MCE-IRL). |
| Behavioral recovery (policy, value, and Q) matters more than a structural parameter vector. | Identified structural parameters are required (use the structural family). |
| A normalization anchor can be fixed before estimation. | The reward normalization cannot be fixed in advance. |
| Counterfactual re-solving under the learned reward is the goal. | Policy-only imitation is sufficient (use BC). |

Deep MCE-IRL occupies the same position as MCE-IRL in the IRL family, with
greater reward capacity at the cost of interpretability. Against GLADIUS, the
distinction is the planning method: Deep MCE-IRL solves the soft Bellman
explicitly each epoch using supplied transitions, while GLADIUS trains Q and
value networks with a Bellman consistency penalty and does not require
transitions to be supplied. Against the structural family (NFXP, CCP, MPEC),
Deep MCE-IRL does not identify a finite parameter vector; it identifies the
anchored reward matrix and the behavior it induces.

## Usage

The [Quick Start](deep_mce_irl/quick_start.md) page gives a complete runnable
fit with exact output. The fitted estimator provides `reward_matrix_`,
`policy_`, `value_`, `simulate()`, and `counterfactual()`.

`counterfactual()` accepts one reward, transition, or action-availability
change. It re-solves the learned reward map without retraining the network.
The [Counterfactuals](deep_mce_irl/counterfactuals.md) page shows the supported
inputs and result object.

## Evidence

Behavioral recovery is measured on a synthetic benchmark with 32 states and
three actions. It uses 160,000 observations, a nonlinear neural reward,
stochastic transitions, and an anchor action. The reward matrix, policy, value
function, Q function, and counterfactual objects are known for this cell.

All four controlled cells converged and passed 36 of 36 checks. The table
shows the primary nonlinear reward cell.

| Metric | Value |
| --- | ---: |
| Policy total variation | 0.0047 |
| Reward normalized RMSE | 0.0436 |
| Value normalized RMSE | 0.0778 |
| Q normalized RMSE | 0.0442 |
| Type A regret (reward shift) | 0.00164 |
| Type B regret (transition change) | 0.00148 |
| Type C regret (action removed) | 0.00191 |

Policy total variation below 0.005 and counterfactual regrets below 0.002
across all three families indicate that the learned reward map reproduces the
demonstrator's behavior and supports counterfactual re-solving with low error.
There is no parameter-recovery table: the reward is identified only up to the
chosen normalization, not as a unique finite vector.

The separate [Simulation Study](deep_mce_irl/validation.md) varies both the
generated panel and neural initialization. All 300 fits converged. Median
reward normalized RMSE was 0.0677, median policy total variation was 0.00850,
and median counterfactual regret was below 0.00474 for each intervention
family. These repeated fits measure stability, not sampling uncertainty.

For the cross-estimator comparison, see the
[bus engine simulation study](../simulation_studies/rust_bus.md) and the
[taxi gridworld simulation study](../simulation_studies/taxi_gridworld.md).

## References

Source papers:

- Ziebart, B. D. (2010). *Modeling Purposeful Adaptive Behavior with the
  Principle of Maximum Causal Entropy*. PhD thesis, Carnegie Mellon
  University. {ref}`reference entry <ziebart-2010>`.
- Wulfmeier, M., Ondruska, P., and Posner, I. (2015). "Maximum Entropy Deep
  Inverse Reinforcement Learning." NIPS Deep Reinforcement Learning Workshop.
  {ref}`reference entry <wulfmeier-2015>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimators.mceirl_neural`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/mceirl_neural.py).
- MCE solver: [`econirl.estimation.mce_irl`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/mce_irl.py).
- Validation runner: [`validation/estimators/deep_mce_irl/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/run.py).
- Results file: [`deep_mce_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl.json).

Pages:

- [Quick Start](deep_mce_irl/quick_start.md)
- [Pre-Estimation Checks](deep_mce_irl/pre_estimation.md)
- [Simulation Study](deep_mce_irl/validation.md)
- [Wulfmeier-Shaped Study](deep_mce_irl/wulfmeier_objectworld.md)
- [Counterfactuals](deep_mce_irl/counterfactuals.md)

```{toctree}
:hidden:

deep_mce_irl/quick_start
deep_mce_irl/pre_estimation
deep_mce_irl/validation
deep_mce_irl/wulfmeier_objectworld
deep_mce_irl/counterfactuals
```
