# AIRL

Adversarial inverse reinforcement learning recovers a reward function from
observed demonstrations by training a structured discriminator that separates
a transferable reward component from a potential-based shaping term. The
estimator follows Fu, Luo, and Levine (2018): a discriminator of constrained
form is trained to distinguish expert transitions from policy-generated ones,
and at the adversarial optimum the reward component tracks the true reward up
to a constant. This package implements the tabular variant; the validated cell
satisfies the state-only reward condition required by the original
disentanglement theorem.

## Source Papers

The estimator follows {ref}`Fu, Luo, and Levine (2018) <fu-2018>`, which
introduces the structured adversarial discriminator and proves the
disentanglement result for state-only rewards in decomposable MDPs.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action,
observed for individual $i$ in period $t$. The discount factor is $\beta$ and
the logit shock scale is $\sigma$. This page uses $\beta$ for the discount
factor; Fu et al. (2018) use $\gamma$. These are the same object. The
transition kernel $F_a(s' \mid s)$ gives the probability of moving from $s$
to $s'$ under action $a$, stored in $(A, S, S)$ orientation. The integrated
value function is $V(s)$, the choice-specific value is $Q(s, a)$, and the
conditional choice probability is $\pi(a \mid s)$. The per-period flow utility
is $u(s, a)$, which in the identification setting equals $g_\theta(s)$
(state-only reward). The reward candidate is $g_\theta$, a function of state
in the validated identification setting. The shaping potential is $h_\phi(s)$,
a learned function of the current state. The discriminator logit is
$f_{\theta,\phi}(s, a, s') = g_\theta(s) + \beta\, h_\phi(s') - h_\phi(s)$,
and the discriminator probability is $D(s, a, s')$.

## Model

The observed data are state, action, and next-state triples
$(s_{it}, a_{it}, s_{i,t+1})$ from a stationary infinite-horizon dynamic
discrete choice model with discount factor $\beta$ and i.i.d. logit taste
shocks. The agent's integrated value function solves the soft Bellman fixed
point:

$$
V(s) = \log \sum_{a} \exp\!\Bigl(u(s, a) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr).
$$

The choice-specific value is:

$$
Q(s, a) = u(s, a) + \beta \sum_{s'} F_a(s' \mid s)\, V(s').
$$

AIRL trains a discriminator whose logit is constrained to decompose as a
reward candidate plus a shaping term:

$$
f_{\theta,\phi}(s, a, s') = g_\theta(s) + \beta\, h_\phi(s') - h_\phi(s).
$$

In the original identification setting of Fu et al. (2018), the reward
candidate is constrained to be state-only: $u(s, a) = g_\theta(s)$ for all
$a$. The shaping potential $h_\phi(s)$ absorbs value-like dynamics terms that
would otherwise contaminate the reward estimate.

At the adversarial optimum, $f^*(s,a,s') = Q^*(s,a) - V^*(s) = A^*(s,a)$
(the soft advantage; Appendix A.4 of Fu et al. 2018). Substituting the
constrained form of $f$ into this identity gives the unnesting step:

$$
g^*(s) + \gamma h^*(s') - h^*(s) = r(s) + \gamma V^*(s') - V^*(s).
$$

This bridges $f^*(s,a,s') = A^*(s,a)$ to the component-level result: by
applying Lemma B.1 (the decomposability lemma) to both sides, the left
side separates into a state term and a next-state term, forcing
$h^*_\phi(s) = V^*(s) + \text{const}$ and $g^*_\theta(s) = r^*(s) +
\text{const}$ (Theorem C.1 of Fu et al. 2018). The shaping term
$\beta h_\phi(s') - h_\phi(s)$ therefore converges to $\gamma V^*(s') -
V^*(s)$, the dynamics-dependent component of the advantage.

The discriminator probability is:

$$
D(s, a, s') = \frac{\exp(f_{\theta,\phi}(s, a, s'))}{\exp(f_{\theta,\phi}(s, a, s')) + \pi(a \mid s)}.
$$

The canonical validation instance is a synthetic state-only cell with 16
states, 4 actions, and 4 linear state features. The oracle reward, policy,
value function, Q function, and Type A, B, and C counterfactual oracles are
constructed before generating the demonstration panel. The estimator sees only
the demonstrations, the transitions, and the reward feature basis.

## Identification

AIRL recovers $g_\theta$ under the following assumptions.

- **State-only reward.** The reward is a function of state alone,
  $g_\theta(s)$, not of the state-action pair. Fu et al. (2018) establish
  the disentanglement and identification result only under this condition.
- **MDP decomposability.** The MDP satisfies the decomposability condition of
  Fu et al. (2018): the optimal Bellman value at the adversarial equilibrium
  separates into a reward term and a continuation-value term, enabling
  $h_\phi$ to absorb the dynamics-dependent component. Formally (Fu et al.
  2018, Def. B.1), the dynamics satisfy decomposability if all states are
  linked, meaning every pair of states can be reached from a common predecessor
  in one step (transitively). Under this condition, Lemma B.1 guarantees that
  any identity $a(s) + b(s') = c(s) + d(s')$ for all $s, s'$ implies
  $a(s) = c(s) + \text{const}$ and $b(s') = d(s') + \text{const}$, enabling
  the $f^* = g^* + \beta h^*(s') - h^*(s)$ decomposition.
- **Adversarial optimum.** The discriminator reaches the point $D = 1/2$
  everywhere, which forces $f_{\theta,\phi}(s,a,s') = \log \pi(a \mid s)$
  and pins $g_\theta$ to the true reward up to a constant. The algebra is:
  at $D = 1/2$, $\exp\{f_{\theta,\phi}\} / (\exp\{f_{\theta,\phi}\} +
  \pi(a \mid s)) = 1/2$, so $\exp\{f_{\theta,\phi}\} = \pi(a \mid s)$,
  giving $f^*(s,a,s') = \log \pi_E(a \mid s) = A^{\pi_E}(s,a)$, the optimum and
  its advantage interpretation both established in Appendix A.4 of Fu et al.
  (2018).
- **Additive separability.** The per-period payoff is the systematic reward
  plus an additive i.i.d. logit shock, which produces the soft Bellman
  equation and the logit policy.
- **Exogenous transitions.** The transition kernel $F_a(s' \mid s)$ is
  supplied externally. The adversarial game conditions on the transitions;
  misspecified transitions contaminate the reward estimate.
- **Feature rank.** For a linear reward parameterization, the state feature
  matrix must have full column rank over the observed state support.

**Theorem** (Fu et al. 2018, Appendix C, Theorem C.1). Under deterministic
dynamics and the decomposability condition, if $g_\theta$ is constrained to
state-only and the discriminator reaches its adversarial optimum, then
$g^*_\theta(s) = r(s) + \text{const}$ and $h^*_\phi(s) = V^*(s) +
\text{const}$.

These hold inside a finite discrete state space, a stationary environment, and
a known fixed discount $\beta$. Given them, $g_\theta$ is identified up to a
constant. The potential-based shaping ambiguity (any
$r'(s,a,s') = r(s,a,s') + \beta h(s') - h(s)$ is behaviorally equivalent
under the original dynamics) is absorbed by $h_\phi$ under the state-only
condition. Identification fails under two modes: (i) a state-only $g_\theta(s)$ cannot
represent the action contrast when the DGP has action-dependent payoffs; (ii)
expanding to $g_\theta(s,a)$ breaks the decomposability of $f^*$ and
$h_\phi$ no longer absorbs the shaping. Details at
[Identification Boundary](airl/identification.md).

## Estimator

The discriminator is trained to distinguish expert transitions from
policy-generated ones via binary cross-entropy. The adversarial objective is:

$$
\max_{\theta,\phi}\;
\mathbb{E}_{(s,a,s') \sim \tau^*}\!\bigl[\log D(s,a,s')\bigr]
+ \mathbb{E}_{(s,a,s') \sim \pi}\!\bigl[\log\bigl(1 - D(s,a,s')\bigr)\bigr].
$$

Equivalently, the discriminator is trained on the log-odds

$$
\ell(s, a, s') = f_{\theta,\phi}(s,a,s') - \log \pi(a \mid s),
$$

maximizing $\log \sigma(\ell)$ on expert transitions and
$\log(1-\sigma(\ell))$ on policy transitions, where $\sigma$ is the logistic
function. This log-odds $\ell$ is also the reward signal for policy
optimization: $\hat{r}(s,a,s') = \log D - \log(1-D) = f_{\theta,\phi}(s,a,s')
- \log \pi(a \mid s)$ (Appendix A.3 of Fu et al. 2018). In the Algorithm
below, step 14 builds $Q_g$ from $g_\text{state}$ rather than $\ell$ directly;
the connection is that $g_\text{state}$ is the state component of $\hat{r}$
after absorbing $h_\phi$ into the value solve. A small $\ell_2$ penalty
$\lambda(\|\theta\|^2 + \|h_\phi\|^2)$ suppresses parameter drift along
directions that leave behavior unchanged.

In the Algorithm, $L$ is the negative of the adversarial objective above;
minimizing $L$ is equivalent to the maximization stated here.

## Algorithm

```text
Algorithm  AIRL (adversarial inverse reinforcement learning, default variant)
Input   panel {(s_it, a_it, s_{i,t+1})}, transitions F_a in (A, S, S) orientation,
        discount beta, max_rounds R, discriminator_steps K
Output  reward table g (state-only), policy pi, value V

1   initialize reward table g[s, a] := 0, shaping potential h[s] := 0
2   initialize pi := uniform over actions
3   compute initial state distribution mu_0 from first-period states in panel
4   for r = 1, ..., R:                                     # outer adversarial loop
5       sample policy transitions (s, a, s') by rolling out pi under F_a
6       for k = 1, ..., K:                                 # discriminator update
7           g_state[s] := mean_a g[s, a]                  # project to state-only subspace
8           f(s,a,s') := g_state[s] + beta h[s'] - h[s]
9           ell(s,a,s') := f(s,a,s') - log pi(a|s)
10          L := -mean_{expert}[log sigma(ell)] - mean_{policy}[log(1 - sigma(ell))]
11               + lambda (||g||^2 + ||h||^2)
12          update (g, h) via Adam on grad_{g,h} L
13      solve V := T_g V via hybrid iteration              # inner loop: policy update
14      Q_g(s,a) := g_state[s] + beta sum_{s'} F_a(s,s') V[s']
15      pi(a|s) := exp(Q_g(s,a)) / sum_b exp(Q_g(s,b))
16      if max|pi_new - pi_old| < convergence_tol and r >= min_rounds: stop
17  return g_state[s] := mean_a g[s, a], pi, V
```

The inner solve in step 13 uses `generator_solver="hybrid"`: value iteration
until near the fixed point, then Newton-Kantorovich steps. The default
`generator_reward="recovered"` re-solves the policy on the raw state reward
$g_\text{state}$; the `"f"` variant re-solves on the shaped score
$g_\text{state}(s) + \beta \mathbb{E}_{s'}[h_\phi(s')] - h_\phi(s)$, which
is the configuration used in the reported state-only study. The `reward_type` argument
controls the parameterization: `"tabular"` (the default above) learns an
$(S, A)$ table and projects it onto the state subspace; `"linear"` with state
features $\phi(s)$ sets $g_\theta(s) = \phi(s)^\top\theta$ directly. The
`reward_arg` controls the scope: `"state"` (default, matching Fu et al. 2018)
enforces the state-only projection; `"state_action"` drops the projection but
loses the disentanglement property.

## System View

AIRL learns from an adversarial comparison between expert transitions and
transitions generated by the current policy. The discriminator has a special
shape: one term is read as reward, and the other term is read as shaping or
continuation value.

```text
Expert transitions
Current policy and generated transitions
        |
        v
Structured discriminator
  reward term + discounted shaping term - current shaping term
        |
        v
Discriminator separates expert from generated transitions
        |
        v
Policy is updated using the discriminator signal
        |
        v
Repeat until the policy and reward signal stabilize
```

The state-only restriction is what lets the reward term carry the original
AIRL transfer claim. If the real payoff differs by action, AIRL-Het or a
structural estimator is the right page to open next.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| Demonstrations come from a discrete dynamic decision problem. | The reward is action-dependent (prefer AIRL-Het or MCE-IRL). |
| The target reward is state-only and the DGP satisfies decomposability. | Structural standard errors or a likelihood-based estimate are required. |
| An adversarial IRL comparison is the research objective. | Counterfactual analysis requires a reward in the same parameterization as the DGP. |
| Transitions are available for policy update and evaluation. | Thin state coverage makes adversarial training unstable. |

AIRL's distinctive aim is a reward that transfers. Fu et al. (2018) show that a
disentangled reward, one that re-optimizes to the right behavior under changed
dynamics, must be a function of state alone. That is the state-only restriction
above, and it is what separates AIRL from its neighbors. GAIL recovers only a
policy: its discriminator converges to one-half everywhere, so it carries no
reward to re-optimize and cannot transfer to new dynamics. MCE-IRL recovers a
reward without an adversarial game, which is more stable in small tabular
environments, but it is not built for the transfer guarantee. AIRL-Het extends
the design to anchored action-dependent rewards for heterogeneous populations,
documented separately.

## Usage

```python
from econirl.estimation import AIRLConfig
from econirl.estimation.adversarial import AIRLEstimator

config = AIRLConfig(
    reward_type="linear",
    reward_arg="state",        # matches Fu et al. (2018) identification setting
    use_shaping=True,
    max_rounds=150,
)
estimator = AIRLEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)         # shape (n_states, n_actions)
print(summary.converged)      # True if policy converged before max_rounds
```

The recovered reward $g_\theta$ is identified only up to a constant; raw
parameter values are not comparable to the data-generating truth. Policy total
variation and counterfactual regret are the appropriate evaluation metrics.
`AIRLEstimator` does not expose a `.counterfactual()` method. Counterfactual
regret is evaluated by re-solving the oracle problem under the intervention and
measuring the welfare cost of the fixed AIRL policy in the changed environment.

```python
# Inspect fitted objects after estimation
policy     = summary.policy            # shape (n_states, n_actions)
value      = summary.value_function    # shape (n_states,)
disc_loss  = summary.metadata["final_disc_loss"]   # inspect before interpreting reward metrics

# Type A, B, and C counterfactual regrets appear in the Evidence section below.
```

The [Quick Start](airl/quick_start.md) page documents the full set of fitted
attributes and the full estimator API.

## Evidence

AIRL is evaluated on a synthetic state-only reward DGP with 16 states and 4
actions that matches the identification conditions of Fu et al. (2018). The
action-dependent boundary, where a state-only reward cannot represent the action
contrast, is covered on the [Identification Boundary](airl/identification.md) page.

The metrics below come from a run with 24,000 observations and 150 training
rounds.

Behavioral recovery against the known oracle:

| Metric | Value |
| --- | ---: |
| Policy total variation | 0.0060 |
| Reward NRMSE | 0.0998 |
| Value NRMSE | 0.1099 |
| Q NRMSE | 0.1201 |

Counterfactual regret, measuring welfare loss from deploying the fixed AIRL
policy in the intervened environment:

| Counterfactual | Regret |
| --- | ---: |
| Type A (reward shift) | 0.0029 |
| Type B (transition change) | 0.0038 |
| Type C (action removed) | 0.0050 |

All three regret values fall below 0.01, consistent with a policy close to the
oracle in the base environment. Reward NRMSE reflects the normalized distance
after projecting both the recovered and oracle rewards onto the state feature
space; the raw adversarial weights are not on a common scale with the oracle.

For cross-estimator comparison, see the
[bus engine simulation study](../simulation_studies/rust_bus.md) and the
[taxi gridworld simulation study](../simulation_studies/taxi_gridworld.md).

## References

Source papers:

- Fu, J., Luo, K., and Levine, S. (2018). "Learning Robust Rewards with
  Adversarial Inverse Reinforcement Learning." *International Conference on
  Learning Representations*. {ref}`reference entry <fu-2018>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.adversarial.airl`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/adversarial/airl.py).
- Neural extension: [`econirl.estimators.neural_airl`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/neural_airl.py).
- Validation runner: [`validation/estimators/airl/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/airl/run.py).
- Results file: [`airl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl.json).

Pages:

- [Quick Start](airl/quick_start.md)
- [Pre-Estimation Checks](airl/pre_estimation.md)
- [Simulation Study](airl/validation.md)
- [Counterfactuals](airl/counterfactuals.md)
- [Identification Boundary](airl/identification.md)

```{toctree}
:hidden:

airl/quick_start
airl/pre_estimation
airl/validation
airl/counterfactuals
airl/identification
```
