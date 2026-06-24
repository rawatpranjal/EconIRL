# Deep MCE-IRL

Deep MCE-IRL recovers a neural reward map from observed choices by matching
the discounted state-action occupancy of an entropy-regularized policy to the
occupancy observed in demonstrations. It uses the same soft Bellman planning
and occupancy-matching objective as MCE-IRL, but the reward is a feedforward
network rather than a dot product with fixed features. The validated object is
the anchored reward matrix and the behavior it induces; raw network weights are
not a structural estimand.

## Source Papers

The estimator draws on {ref}`Ziebart (2010) <ziebart-2010>`, which establishes
the maximum causal entropy IRL framework, the soft Bellman planning operator,
and the occupancy-matching gradient. {ref}`Wulfmeier, Ondruska, and Posner
(2015) <wulfmeier-2015>` introduced the neural reward parameterization into
this framework, noting that the occupancy-mismatch gradient flows through the
reward matrix entry by entry and can be backpropagated through any
differentiable reward network.

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
current policy is $D_\pi(s, a)$. $N$ denotes the total number of
agent-period observations. The initial-state distribution is $\rho_0(s)$,
giving the probability of starting in state $s$.

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

Deep MCE-IRL identifies the anchored reward map $r_\eta$ under the following
conditions.

- **Known transitions.** The transition kernel $P(s' \mid s, a)$ is supplied
  externally and is not estimated jointly with the reward.
- **Causal separability.** Per-period payoffs are the systematic reward plus
  an additive i.i.d. Type-I extreme-value shock with scale $\sigma = 1$,
  drawn independently across choices. The agent acts before observing future
  randomness, so the soft Bellman operator respects the causal direction of
  time in the decision problem.
- **Reward normalization.** A neural reward map is identified only up to an
  action-independent additive function of the state: two maps that differ by
  such a shift induce identical policies. An anchor action whose reward column
  is set to zero for all states, or an absorbing state whose reward row is set
  to zero, removes this indeterminacy. The package enforces anchor
  normalization by pinning action $a_0$:

  $$r_\eta(s, a_0) = 0 \quad \text{for all } s.$$

  Reward-map comparisons are meaningful only under the same normalization.
- **State encoding supplied.** The state encoder $x(s)$ is supplied
  externally. The reward map is identified relative to the chosen encoding;
  a different encoding produces a reward map on a different domain.
- **Neural weight non-identification.** Multiple network parameter vectors
  $\eta$ can represent the same reward matrix. The identified object is the
  reward matrix under the chosen normalization, not the raw network weights.

These conditions hold inside a finite discrete state space with a stationary
environment and a fixed discount factor $\beta$. Given them, the anchored
reward matrix and the induced policy, value, and Q functions are identified.
Identification weakens under thin state-action coverage, an inconsistent
normalization anchor, or a poorly conditioned state encoding. Finite projected
parameters are identified only when the feature projection is
well-conditioned; see the
[Pre-Estimation Checks](deep_mce_irl/pre_estimation.md) page.

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

The empirical discounted occupancy from demonstrations is

$$
D_\text{data}(s, a)
= \frac{1}{N} \sum_{i,t} \beta^t \,\mathbf{1}[s_{it} = s,\; a_{it} = a],
$$

where $N$ is the total number of agent-period observations. The discounted
$\beta^t$ weighting follows the DDC convention; Wulfmeier (2015) and Ziebart
(2008) state the gradient for undiscounted visitation counts. The same
chain-rule argument extends to the discounted case. Define the discounted
model occupancy $D_\pi^\beta(s,a) = (1-\beta)\sum_{t=0}^\infty \beta^t
P(s_t=s,\, a_t=a \mid \pi,\rho_0)$. The MaxEnt log-likelihood is the data
reward minus the log-partition term $\log Z^\beta$, and the partition gradient
is the model occupancy,

$$
\frac{\partial \log Z^\beta}{\partial r(s,a)} = D_\pi^\beta(s,a),
$$

so the occupancy-matching gradient identity becomes

$$
\frac{\partial L}{\partial r(s,a)}
= D_\text{data}(s,a) - D_\pi^\beta(s,a).
$$

The model occupancy $D_\pi(s, a)$ is computed by the discounted forward pass:

$$
D_\pi(s) = \rho_0(s) + \beta \sum_{s', a} D_\pi(s', a)\, P(s \mid s', a),
\qquad
D_\pi(s, a) = \pi(a \mid s)\, D_\pi(s),
$$

where $\rho_0$ is the initial-state distribution. Because $D_\pi$ depends on
the policy, which depends on the soft Bellman solution, differentiating
through the full solve is expensive. The implementation instead uses a
surrogate loss whose gradient equals the chain-rule gradient of the occupancy
mismatch:

$$
L_\text{surrogate}(\eta)
= \sum_{s,a} r_\eta(s, a)\cdot
  \bigl(D_\pi(s, a) - D_\text{data}(s, a)\bigr).
$$

Minimizing $L_\text{surrogate}$ over $\eta$ is equivalent to maximizing the
MCE log-likelihood; the sign convention here is for gradient descent (model
minus data), matching Wulfmeier (2015) eq. 11 up to sign. The chain-rule
decomposition (Wulfmeier 2015, eqs. 10–11) shows that

$$
\nabla_\eta L_\text{surrogate}
= \frac{\partial L}{\partial r} \cdot \frac{\partial r}{\partial \eta}
= \sum_{s,a} \bigl(D_\pi(s, a) - D_\text{data}(s, a)\bigr)
  \frac{\partial r_\eta(s, a)}{\partial \eta},
$$

where $\partial L / \partial r(s,a) = D_\pi(s,a) - D_\text{data}(s,a)$ is the
occupancy mismatch at each $(s,a)$ cell (Ziebart 2008 for the occupancy-matching
gradient identity) and $\partial r_\eta(s,a)/\partial \eta$ is obtained by
backpropagating through the reward network $f_\eta$.

By the occupancy-matching identity (Ziebart 2008), the gradient of the MCE
log-likelihood with respect to $r(s,a)$ equals the negative occupancy mismatch,
so the surrogate gradient equals the negative log-likelihood gradient:

$$
\nabla_\eta L_\text{surrogate}(\eta)
= -\nabla_\eta L(\eta).
$$

Minimizing $L_\text{surrogate}$ by gradient descent is therefore equivalent to
maximizing the MCE log-likelihood; no additional approximation is involved.

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
5   set best_loss = infinity,  patience_counter = 0
6   for epoch = 1, ..., max_epochs                 # outer loop: AdamW descent
7       R(s,a) := f_eta([x(s), e(a)])  for all (s,a)   # neural reward matrix
8       set R(s, a_0) := 0             # enforce anchor normalization
9       solve V, pi via hybrid soft value iteration (R, P)     # inner Bellman
10      compute D_pi(s,a) via discounted forward pass using pi and P
11      grad_R(s,a) := D_pi(s,a) - D_data(s,a)         # occupancy mismatch
12      loss := sum_{s,a} R(s,a) * grad_R(s,a)          # surrogate loss
13      backpropagate grad_R through f_eta;  mask gradients for R(s,a_0) to zero;  AdamW step
14      if loss < best_loss - tol:  update checkpoint;  patience_counter := 0
15      else:  patience_counter := patience_counter + 1
16      if patience_counter >= patience:  break           # early stopping
17  restore best checkpoint
18  re-solve V, pi at best R via hybrid soft value iteration
19  return R_hat := R(s,a), pi, V
```

Gradients with respect to entries $R(s, a_0)$ are masked to zero before the
AdamW step (step 13), so the anchor normalization is enforced throughout
training, not only at inference.

The inner solve in steps 9 and 18 defaults to `inner_solver="hybrid"`:
successive approximation while the Bellman residual is above a switch
tolerance, then Newton-Kantorovich steps near the fixed point. The alternative
`inner_solver="value"` uses plain value iteration throughout, which is
robust from any start but converges more slowly near the solution. The outer
optimizer is AdamW with a global gradient-norm clip of 1.0, implemented via
Equinox and Optax. The implementation lives in
`econirl.estimators.mceirl_neural`; the lower-level MCE solver is in
`econirl.estimation.mce_irl`.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| Transitions are known or supplied. | Transitions must be estimated jointly. |
| The reward is nonlinear in the available state encodings. | A linear reward table is adequate (use MCE-IRL). |
| Behavioral recovery --- policy, value, and Q --- matters more than a structural parameter vector. | Identified structural parameters are required (use the structural family). |
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

```python
from econirl.estimators import MCEIRLNeural

model = MCEIRLNeural(
    n_states=32, n_actions=3, discount=0.95,
    reward_type="state_action", anchor_action=0,
)
model.fit(
    data=df, state="state", action="action", id="agent_id",
    transitions=transitions,
)

print(model.reward_.shape)   # (32, 3) -- anchored reward matrix R(s,a)
print(model.policy_.shape)   # (32, 3) -- choice probabilities pi(a|s)
print(model.summary())
```

Counterfactual analysis re-solves the model under a changed environment
using the learned reward matrix. For a Type A intervention (reward shift),
the anchored reward matrix is modified and the soft Bellman is re-solved:

```python
import numpy as np
from econirl.core.solvers import hybrid_iteration
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.types import DDCProblem

R_cf = model.reward_matrix_.copy()
R_cf[:, 1] += 0.5        # raise the reward for action 1 by 0.5

problem = DDCProblem(
    num_states=32, num_actions=3,
    discount_factor=0.95, scale_parameter=1.0,
)
bellman = SoftBellmanOperator(problem=problem, transitions=transitions)
result = hybrid_iteration(bellman, R_cf)
print(result.policy)     # new choice probabilities under the intervention
```

The [Quick Start](deep_mce_irl/quick_start.md) page documents the full set of
fitted attributes, including `reward_matrix_`, `value_`, and the optional
feature projection interface.

## Evidence

Behavioral recovery is measured on the `deep_mce_neural_reward` synthetic
cell, which has 32 states, 3 actions, 160,000 observations, a nonlinear
neural reward, stochastic transitions, and an anchor action that normalizes
the reward map. The oracle reward matrix, policy, value function, Q function,
and counterfactual objects are all known for this cell, so every figure below
is compared against the oracle.

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
- Lower-level MCE solver: [`econirl.estimation.mce_irl`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/mce_irl.py).
- Validation runner: [`validation/estimators/deep_mce_irl/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/run.py).
- Results file: [`deep_mce_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl.json).

Pages:

- [Quick Start](deep_mce_irl/quick_start.md)
- [Pre-Estimation Checks](deep_mce_irl/pre_estimation.md)
- [Simulation Study](deep_mce_irl/validation.md)
- [Counterfactuals](deep_mce_irl/counterfactuals.md)

```{toctree}
:hidden:

deep_mce_irl/quick_start
deep_mce_irl/pre_estimation
deep_mce_irl/validation
deep_mce_irl/counterfactuals
```
