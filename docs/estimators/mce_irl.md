# MCE-IRL

## Important Links

- [Quick start](mce_irl/quick_start.md)
- [Pre-estimation checks](mce_irl/pre_estimation.md)
- [Simulation Study](mce_irl/validation.md)
- [Counterfactuals](mce_irl/counterfactuals.md)
- [Applied notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/mce-irl/mce_irl_applied_workflow.ipynb)

Maximum causal entropy inverse reinforcement learning recovers reward
parameters from demonstrated state-action trajectories by matching discounted
feature expectations under a soft-optimal causal policy. For each candidate
reward, the estimator solves the soft dynamic program, propagates the implied
state distribution forward, and computes the model feature moments. It updates
the parameters until the model moments equal the expert moments.
Counterfactuals are meaningful only through the fitted MDP primitives.

Read this page when demonstrations, not a structural likelihood, define the
problem. The estimated object is a reward inside the supplied feature basis and
normalization.

## Source Papers

The estimator follows {ref}`Ziebart et al. (2008) <ziebart-2008>`, which
introduces maximum-entropy inverse reinforcement learning through feature-count
matching, and {ref}`Ziebart (2010) <ziebart-2010>`, which formulates the
maximum causal entropy objective. Causal entropy conditions each action on
information available when the choice is made. This includes the current state
and continuation values. It excludes subsequently realized states. Trajectory
maximum-entropy IRL instead scores the whole trajectory. Its entropy therefore
depends on the transition dynamics and can favor actions with uncertain
outcomes. The two formulations coincide under deterministic dynamics but differ
under stochastic dynamics. The causal form has the same soft choice structure
as logit dynamic discrete choice. MCE-IRL estimates reward parameters through
feature matching rather than a conditional likelihood.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action,
observed for individual $i$ in period $t$. The vector $\phi(s, a)$ collects
the action-dependent reward features and $\theta$ the reward parameters to be
estimated. The subscript $k$ indexes the $k$-th component of $\phi$, so
$\phi_k(s, a)$ denotes the $k$-th feature value at $(s, a)$. The discount
factor is $\beta$ and the logit shock scale is $\sigma$. Setting $\sigma = 1$
gives the unit-scale convention used by Ziebart (2010). Keeping $\sigma$
explicit makes the connection to logit dynamic discrete choice clear. The
transition kernel $P_a(s, s')$ gives the probability of
moving to $s'$ from $s$ under action $a$. A dense kernel uses $(A, S, S)$
orientation. A deterministic system stores one successor in
`next_state[s, a]` and one legality flag in `valid_action[s, a]`.
The initial state distribution is $\rho_0(s)$. The soft value function is
$V_\theta(s)$, the choice-specific value is $Q_\theta(s, a)$, and the causal
policy is $\pi_\theta(a \mid s)$. The empirical discounted expert occupancy is
$D_E(s, a)$ and the model occupancy is $D_\theta(s, a)$. The expert and model
feature moments are $\mu_E$ and $\mu_\theta$.

## Model

The observed data are state, action, and next-state trajectories
$(s_{it}, a_{it}, s_{i,t+1})$. The reward is linear in the action-dependent
features:

$$
r_\theta(s, a) = \phi(s, a)^\top \theta.
$$

The choice-specific value satisfies:

$$
Q_\theta(s, a) = r_\theta(s, a) + \beta \sum_{s'} P_a(s, s') V_\theta(s').
$$

The soft value function solves:

$$
V_\theta(s)
= \sigma \log \sum_a \exp\!\left(\frac{Q_\theta(s, a)}{\sigma}\right).
$$

The causal policy is the softmax of the choice-specific values:

$$
\pi_\theta(a \mid s)
= \frac{\exp(Q_\theta(s, a) / \sigma)}
       {\sum_b \exp(Q_\theta(s, b) / \sigma)}.
$$

Action probabilities at time $t$ depend on the current state and continuation
values, not on future realized states. This causal structure connects MCE-IRL
to logit dynamic discrete choice: both use the same soft choice form, but
MCE-IRL estimates the reward through feature moments rather than through a
conditional likelihood alone.

For a finite horizon, the estimator keeps a separate value and policy for each
period. A terminal state is absorbing and has zero continuation value. The
discount factor may equal one only in this finite-horizon case.

## Shared Tasks

Route data often contain many origins and destinations over one road graph.
`MCEIRLTask` represents each problem. A task supplies a task identifier, a
start distribution, terminal states, a finite horizon, and optional active
states and legal actions.

The compiler builds compact disjoint views for the dynamic program. It does
not duplicate a global dense transition tensor. Reward features and parameters
remain shared across tasks. This matches the road-choice structure in Ziebart
et al. (2008), where destinations define different MDPs and one reward vector
explains all routes.

Each demonstration carries a task identifier. Its states must remain inside
the active subgraph. Its action and next state must agree with the fixed
transition system.

## Identification

Identification requires feature matching to recover the intended reward
representation, not merely reproduce observed behavior.

MCE-IRL identifies a reward representation under the following assumptions.

- **Known transitions.** The transition kernel $P_a(s, s')$ is supplied or
  estimated outside the estimator. It does not depend on the reward parameters.
- **Causal behavioral model.** The agent's policy is the soft-optimal policy
  of the maximum causal entropy objective. The action distribution at each
  state follows the softmax of the choice-specific values from the soft Bellman
  recursion.
- **Additive linear reward.** The reward is linear in the supplied feature
  matrix: $r_\theta(s, a) = \phi(s, a)^\top \theta$. Structural
  counterfactuals require this parametric form.
- **Reward normalization.** The reward is identified only up to
  transformations that leave behavior unchanged, including additive constants
  and reward shaping. A normalization anchor must be applied consistently when
  comparing estimated and reference rewards.
- **Action-contrast identification.** After normalization, the feature-moment
  Jacobian with respect to $\theta$ must have full column rank. Raw feature
  rank is not enough. Features that are constant across feasible actions can
  difference out of choice probabilities and leave reward directions
  unidentified.
- **Sufficient action support.** Each action must have enough observed support
  for the occupancy comparison. States with only one feasible action, or rare
  actions in the data, leave the corresponding reward directions weakly
  pinned.
- **Consistent encoding.** Observations must be encoded in the same
  state-action indexing system as the transition tensor.

These hold inside a finite discrete state space with a stationary environment
and a known discount factor $\beta$. When the normalized moment map is
one-to-one, the condition $\mu_E = \mu_\theta$ determines $\theta$ within the
supplied feature basis. A full-rank local Jacobian supports local
identification. It does not by itself prove global uniqueness. Identification
weakens with deficient action contrasts, thin action support, or an invalid
normalization.

## Estimator

MCE-IRL matches discounted feature expectations. The empirical and model
feature moments are:

$$
\mu_E = \sum_{s,a} D_E(s, a)\,\phi(s, a),
\qquad
\mu_\theta = \sum_{s,a} D_\theta(s, a)\,\phi(s, a).
$$

The estimator solves the moment condition:

$$
\mu_E - \mu_\theta = 0.
$$

Equivalently, this is the stationarity condition of the causal-entropy dual
objective. The primal problem is

$$
\max_{\pi \text{ causal}} H_\text{causal}(\pi)
\quad \text{s.t.} \quad
\mathbb{E}_\pi[\phi(s,a)] = \mu_E,
$$

where $H_\text{causal}(\pi)$ is the causal entropy of the policy. Introducing
$\theta$ as the Lagrange multiplier on the feature-matching constraint gives
the concave dual target

$$
L(\theta) = \min_{\pi \text{ causal}}
  \left[
    \theta \cdot (\mu_E - \mu_\pi)
    - H_\text{causal}(\pi)
  \right].
$$

Its gradient is $\nabla_\theta L(\theta) = \mu_E - \mu_\theta$
(Ziebart, 2010, ch. 3). The inner minimization is equivalent to maximizing
causal entropy plus expected reward. Its dynamic-programming solution is the
soft-optimal policy, which is the softmax of $Q_\theta / \sigma$ derived above.
Differentiating the resulting log policy gives the score below. Since the
policy is the softmax of $Q_\theta / \sigma$, the score has the logit form

$$
\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_k}
= \frac{1}{\sigma}\left(
    \frac{\partial Q_\theta(s, a)}{\partial \theta_k}
    - \sum_b \pi_\theta(b \mid s)\,\frac{\partial Q_\theta(s, b)}{\partial \theta_k}
  \right),
$$

where the $Q$-gradient carries a continuation term through the value function,

$$
\frac{\partial Q_\theta(s, a)}{\partial \theta_k}
= \phi_k(s, a)
  + \beta \sum_{s'} P_a(s, s')\,\frac{\partial V_\theta(s')}{\partial \theta_k}.
$$

For the causal-entropy dual, summing the occupancy recursion under the expert
and model distributions gives the feature-expectation gradient
(Ziebart, 2010, §3.4):

$$
\nabla_\theta L(\theta) = \mu_E - \mu_\theta.
$$

The infinite-horizon conditional log-likelihood fit uses the related score
described below.

For an infinite-horizon fit, the default L-BFGS-B path maximizes the
conditional log likelihood under $\pi_\theta$. Its gradient uses implicit
differentiation through the soft Bellman fixed point. A finite-horizon fit
instead uses backward induction and solves the feature-moment condition
directly. The infinite-horizon score differentiates through the value
function via:

$$
(I - \beta P_\pi)\frac{\partial V}{\partial \theta_k}
= \sum_a \pi_\theta(a \mid s)\,\phi_k(s, a),
$$

where $P_\pi = \sum_a \operatorname{diag}(\pi_\theta(\cdot, a)) P_a$ is the
policy-weighted transition matrix.

The model state occupancy $D_\theta(s)$ required for $\mu_\theta$ is computed
by a forward pass (Ziebart, 2010, Algorithm 1):

$$
D_\theta(s) = \rho_0(s)
  + \beta \sum_{s', a} D_\theta(s')\,\pi_\theta(a \mid s')\,P_a(s', s),
$$

or in matrix form $D_\theta = \rho_0 + \beta P_\pi^\top D_\theta$, solved by
fixed-point iteration. The implementation normalizes this discounted
occupancy to sum to one, matching the normalized empirical discounted
occupancy. The state-action occupancy is then
$D_\theta(s, a) = D_\theta(s)\,\pi_\theta(a \mid s)$, from which
$\mu_\theta = \sum_{s,a} D_\theta(s, a)\,\phi(s, a)$.

The final gradient of the log-likelihood with respect to $\theta_k$ is

$$
\frac{\partial \mathcal{L}}{\partial \theta_k}
= \frac{1}{\sigma} \sum_t
  \Bigl[
    dQ_k(s_t, a_t)
    - \sum_a \pi_\theta(a \mid s_t)\,dQ_k(s_t, a)
  \Bigr],
$$

where $dQ_k(s, a) = \phi_k(s, a) + \beta (P_a\, dV_k)(s)$ and $dV_k$ solves
the implicit-differentiation system above. This step connects the implicit
differentiation to the gradient used in L-BFGS-B. The resulting gradient has the
same logit form as the structural conditional likelihood score.

## Algorithm

```text
Algorithm  MCE-IRL
Input   panel {(s_it, a_it, s_{i,t+1})}, features phi, transitions P,
        discount beta, logit scale sigma
Output  theta_hat, policy pi, value V

1   compute expert feature moments mu_E from the demonstration occupancy
2   obtain rho_0 from the data or the supplied task start distributions
3   initialize theta
4   repeat
5       r_theta(s, a) := phi(s, a)' theta
6       solve the soft dynamic program             # hybrid fixed point or backward induction
7       Q_theta(s, a) := r_theta(s, a) + beta * sum_{s'} P_a(s, s') V_theta(s')
8       pi_theta(a | s) := exp(Q_theta(s, a)/sigma) / sum_b exp(Q_theta(s, b)/sigma)
9       compute model feature moments mu_theta
10      residual := mu_E - mu_theta
11      update theta
12  until the optimizer and residual checks pass
13  verify the Bellman and occupancy checks
14  return theta_hat, pi_theta, V_theta
```

For an infinite horizon, the inner solve in step 6 defaults to
`inner_solver="hybrid"`: value iteration
(contraction) while far from the fixed point, then Newton-Kantorovich steps
near the solution. Two pure variants are also available. `"value"` (successive
approximation) converges linearly and is robust from any start. `"policy"`
(policy iteration with matrix-inversion evaluation) converges faster near the
solution but requires a good starting point.

Finite-horizon fits use backward induction and retain every period's policy
and value function. With `optimizer="L-BFGS-B"`, `"BFGS"`, or `"root"`, they
solve the feature stationarity equation with the HYBR root method. The
`"gradient"` path instead uses Adam or SGD. Infinite-horizon dense fits use
L-BFGS-B with implicit differentiation by default.

A fit reports convergence only when the optimizer, stationarity residual,
occupancy residual when applicable, and Bellman residual all pass. The
`termination_reason_` attribute identifies the first failed check.

## System View

MCE-IRL starts from demonstrations rather than a structural likelihood. It asks
which reward makes a soft-optimal agent visit the same state-action features as
the expert.

```text
Expert demonstrations
Known transition model, reward features, discount factor
        |
        v
Compute expert feature moments from observed behavior
        |
        v
Try one candidate reward parameter theta
        |
        v
Solve the soft dynamic program under that reward
        |
        v
Compute the model's feature moments
        |
        v
Update theta until model moments match expert moments
```

The reward is identified only inside the supplied feature span and
normalization. If the features omit the real action contrast, the estimator can
fit behavior without recovering the intended reward.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| Demonstrations come from a discrete sequential decision problem. | A structural disturbance model beyond soft choice is required. |
| Transitions are known or can be supplied. | Transition estimation is the main modeling challenge. |
| Reward features are supplied and action-dependent. | Reward features are unknown or require a neural representation. |
| The behavioral model is maximum causal entropy. | The target is deterministic control without entropy regularization. |
| Normalized reward comparisons, policy recovery, and counterfactual policy changes are the goals. | Only fitted conditional choice probabilities are required. |

MCE-IRL is the reference entropy IRL estimator for tabular discrete choice.
The structural estimators (NFXP, CCP, MPEC, NNES, TD-CCP) target the same
reward through likelihood or estimating-equation paths and report standard
errors for $\theta$. Neural MCE-IRL keeps the causal-entropy objective but
replaces the tabular feature basis with a neural reward map.

## Usage

The [Quick Start](mce_irl/quick_start.md) gives a complete deterministic,
multi-task example with exact output. Use a dense `(A, S, S)` tensor for a
small stochastic MDP. Use `DeterministicTransitions` for a large sparse system.

The fitted model exposes shared reward parameters, standard errors,
period-specific policies, task-specific policy views, simulation, and
counterfactual re-solving. `diagnostics_` records the rank, support, and
transition checks used by the fit. `capabilities_` states which fitted
operations are available. `summary()` reports the data, model, fit, outcome,
uncertainty, and interpretation limits in one fixed layout. The
[Pre-Estimation Checks](mce_irl/pre_estimation.md) page covers rank, support,
transition alignment, terminal states, and task membership.

## Evidence

The repeated-run inference study completes 300 independent fits. The 95
percent intervals cover the true reward parameter in 96.0 percent of runs.
The asymptotic standard error is 1.057 times the trajectory-bootstrap standard
error computed from 200 resamples. All fits pass the joint convergence checks.

A separate calibration resamples whole individual trajectories. It completes
50 independently generated panels with 99 bootstrap draws per panel. All
4,950 draws succeed. The percentile intervals cover the true parameter in
92.0 percent of panels. Their mean width is 0.383.

A separate generated road study constructs arrays with 302,500 states,
907,500 valid deterministic state-action links, and 22 raw reward features. The
fitted problem contains 64 destination tasks and 7,552 compiled states. It
evaluates 7,403 held-out routes. This checks paper-scale sparse storage and
shared-reward task compilation. It does not reproduce the Pittsburgh graph,
data, or Table 1 results. See [Simulation Study](mce_irl/validation.md) for
the paper values and generated results.

Behavioral recovery is measured on a synthetic benchmark with 25 states, 3
actions, and 8 action-dependent reward features. The reward, transitions, policy,
value, Q functions, and counterfactual oracles are fully specified before any data
are generated. The estimator sees only the 300,000 generated observations, the
transition tensor, and the feature matrix. The root feature-matching path reaches a
solution in 25 iterations.

Behavioral fit against the known oracle policy:

| Metric | Value |
| --- | ---: |
| Policy total variation | 0.00698 |
| Value RMSE | 0.0319 |
| Type A regret (reward shift) | 0.000433 |
| Type B regret (transition change) | 0.000410 |
| Type C regret (action removed) | 9.44e-05 |

Counterfactual recovery under three perturbation families:

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A (reward shift) | 0.006456 | 0.000742 | 0.000433 |
| Type B (transition change) | 0.006284 | 0.000523 | 0.000410 |
| Type C (action removed) | 0.004211 | 0.000145 | 9.44e-05 |

These results are local to the known simulation environment. They depend on the
same transition law, support, reward representation, and policy-response
assumptions used in fitting. For the
cross-estimator comparison on multiple dynamic choice problems, see the
[bus engine simulation study](../simulation_studies/rust_bus.md).

## References

Source papers:

- Ziebart, B. D., Maas, A. L., Bagnell, J. A., and Dey, A. K. (2008). Maximum
  Entropy Inverse Reinforcement Learning. _Proceedings of the 23rd AAAI
  Conference on Artificial Intelligence_, 1433-1438.
  {ref}`reference entry <ziebart-2008>`.
- Ziebart, B. D. (2010). _Modeling Purposeful Adaptive Behavior with the
  Principle of Maximum Causal Entropy_. PhD thesis, Carnegie Mellon University.
  {ref}`reference entry <ziebart-2010>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.mce_irl`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/mce_irl.py).
- sklearn wrapper: [`econirl.MCEIRL`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/mce_irl.py).
- Validation runner: [`validation/estimators/mce_irl/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/mce_irl/run.py).
- Results file: [`mce_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mce_irl.json).
- Inference runner: [`validation/estimators/mce_irl/ready.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/mce_irl/ready.py).
- Bootstrap calibration: [`validation/estimators/mce_irl/bootstrap_calibration.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/mce_irl/bootstrap_calibration.py).
- Applied notebook: [`mce_irl_applied_workflow.ipynb`](https://github.com/rawatpranjal/EconIRL/blob/main/examples/mce-irl/mce_irl_applied_workflow.ipynb).
- Road generator: [`validation/estimators/mce_irl/ziebart_road_synthetic.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/mce_irl/ziebart_road_synthetic.py).

Pages:

- [Quick Start](mce_irl/quick_start.md)
- [Pre-Estimation Checks](mce_irl/pre_estimation.md)
- [Simulation Study](mce_irl/validation.md)
- [Counterfactuals](mce_irl/counterfactuals.md)
- [Bus Engine Wiring Example](mce_irl/rust_bus.md)

```{toctree}
:hidden:

mce_irl/quick_start
mce_irl/pre_estimation
mce_irl/validation
mce_irl/counterfactuals
mce_irl/rust_bus
```
