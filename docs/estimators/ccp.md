# CCP

Conditional choice probability estimation recovers primitive reward parameters from
tabular dynamic discrete choice data by replacing repeated Bellman fixed-point
computations with a single-step matrix inversion. The first stage estimates empirical
conditional choice probabilities from state-action frequencies; the Hotz-Miller
inversion theorem converts those probabilities into continuation values without solving
the Bellman equation. The Aguirregabiria-Mira nested pseudo-likelihood extension
iterates the inversion and the pseudo-likelihood to refine the structural estimate
toward full maximum likelihood efficiency.

## Source Papers

The estimator follows {ref}`Hotz and Miller (1993) <hotz-miller-1993>`, which
introduces the conditional-choice-probability inversion theorem for dynamic discrete
choice models. {ref}`Aguirregabiria and Mira (2002) <aguirregabiria-mira-2002>`
introduce the nested pseudo-likelihood algorithm that iterates the Hotz-Miller step
to approach full maximum likelihood efficiency.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action, observed for
individual $i$ in period $t$. The vector $\phi(s, a)$ collects the known reward
features and $\theta$ the reward parameters to be estimated. The discount factor is
$\beta$ and the logit shock scale is $\sigma$. The transition kernel $P_a(s, s')$
gives the probability of moving to $s'$ from $s$ under action $a$, stored in
$(A, S, S)$ orientation. The first-stage empirical conditional choice probability is
$\hat\pi(a \mid s)$, estimated from state-action frequencies. The Euler-Mascheroni
constant $\gamma \approx 0.5772$ enters the logit emax correction. The
policy-weighted transition matrix $F_{\hat\pi}$, augmented feature weights
$W_\phi(s)$ and $W_e(s)$, and the pseudo choice-specific value
$\tilde{Q}_\theta(s, a;\hat\pi)$ are defined in the Model section below.

## Model

The observed data are state, action, and next-state trajectories
$(s_{it}, a_{it}, s_{i,t+1})$. The flow payoff is linear in the features:

$$
u_\theta(s, a) = \phi(s, a)^\top \theta.
$$

Given a first-stage policy $\hat\pi$, define the policy-weighted transition matrix
and the logit emax correction:

$$
F_{\hat\pi}(s, s') = \sum_a \hat\pi(a \mid s)\, P_a(s, s'),
\qquad
e_{\hat\pi}(s, a) = \gamma - \log \hat\pi(a \mid s).
$$

The Hotz-Miller inversion writes the integrated value under $\hat\pi$ as a matrix
resolve:

$$
\bar{V}_{\hat\pi}
= (I - \beta F_{\hat\pi})^{-1}
  \sum_a \hat\pi(a \mid s)
  \bigl\{u_\theta(s, a) + e_{\hat\pi}(s, a)\bigr\}.
$$

For linear rewards, this separates into parameter-dependent and parameter-free parts:

$$
\bar{V}_{\hat\pi}(s) = W_\phi(s)^\top \theta + W_e(s),
$$

where:

$$
W_\phi = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\, \phi(s, a),
\qquad
W_e = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\, e_{\hat\pi}(s, a).
$$

The pseudo choice-specific value combines the flow utility with the discounted
continuation:

$$
\tilde{Q}_\theta(s, a; \hat\pi)
= \phi(s, a)^\top \theta
+ \beta \sum_{s'} P_a(s, s')
  \bigl\{W_\phi(s')^\top \theta + W_e(s')\bigr\}
= \tilde{z}(s,a)^\top \theta + \tilde{e}(s,a),
$$

where $\tilde{z}(s,a) = \phi(s,a) + \beta \sum_{s'} P_a(s,s') W_\phi(s')$ and
$\tilde{e}(s,a) = \beta \sum_{s'} P_a(s,s') W_e(s')$ depend on $\hat\pi$ but not on
$\theta$. One factorization of $(I - \beta F_{\hat\pi})$ per NPL step replaces the
repeated per-evaluation Bellman solves that NFXP pays at every likelihood call.

## Identification

CCP point-identifies the reward parameters $\theta$ under the following assumptions
and support requirements.

- **Conditional Independence (CI).** The observed state transition is Markov in the
  current state and action and does not depend on the current logit shock.
- **Additive Separability (AS).** The per-period payoff is the systematic reward plus
  an additive choice-specific shock drawn independently across choices as Type-I
  extreme value with fixed scale $\sigma$.
- **Exogenous Transitions.** The transition kernel $P_a(s, s')$ is supplied or
  estimated in a first stage, outside the payoff model.
- **Reward Normalization.** The reward level and scale need an anchor. An exit or
  absorbing action with payoff fixed to zero pins the level, and the logit scale
  $\sigma$ is held fixed.
- **Action-Dependent Feature Rank.** The reward features must vary across actions and
  have full column rank. State-only features copied across actions collapse the action
  contrasts and leave $\theta$ unidentified.
- **Action Support.** Every action must have positive empirical mass in each state
  where it is relevant. Near-zero conditional choice probabilities make the emax
  correction $\gamma - \log \hat\pi(a \mid s)$ numerically unstable.
- **State Coverage.** The first-stage policy is estimated nonparametrically by state.
  Sparsely observed states introduce approximation error into the inversion and weaken
  identification of the continuation-value terms.

These hold inside a finite discrete state space, a stationary environment with
expected-utility maximization, and a known, fixed discount factor $\beta$. Given them,
$\theta$ is point-identified. Identification weakens under thin action support, an
invalid normalization, rank-deficient action-contrast features, or sparsely observed
states.

## Estimator

CCP maximizes the pseudo log-likelihood with the policy $\hat\pi^{k-1}$ held fixed
during each inner optimization:

$$
\hat{\theta}_k
= \arg\max_\theta
  \sum_{i,t}
  \log \tilde\pi_\theta(a_{it} \mid s_{it}; \hat\pi^{k-1}),
$$

where the pseudo choice probability follows the logit rule applied to the augmented
features:

$$
\tilde\pi_\theta(a \mid s; \hat\pi)
= \frac{\exp(\tilde{Q}_\theta(s, a; \hat\pi)/\sigma)}
       {\sum_b \exp(\tilde{Q}_\theta(s, b; \hat\pi)/\sigma)}.
$$

Because $\tilde{z}(s,a)$ and $\tilde{e}(s,a)$ do not depend on $\theta$, the
gradient is a closed-form logit score and does not propagate through the inversion:

$$
\psi_i(\theta)
= \frac{1}{\sigma}
  \left[
    \tilde{z}(s_i, a_i)
    - \sum_a \tilde\pi_\theta(a \mid s_i; \hat\pi^{k-1})\, \tilde{z}(s_i, a)
  \right].
$$

Standard errors are computed from the full Bellman-constrained likelihood Hessian,
evaluated numerically at the converged estimate, and from per-observation gradients
for robust sandwich inference. This keeps the fitted summary compatible with the
shared inference interface used by NFXP.

## Algorithm

```text
Algorithm  CCP/NPL (conditional choice probability, K-step nested pseudo-likelihood)
Input   panel {(s_it, a_it, s_{i,t+1})}, features phi, transitions P,
        discount beta, logit scale sigma, NPL steps K
Output  theta_hat, standard errors, policy pi-hat, value V

1   estimate pi-hat(a | s) from state-action frequencies      # first-stage CCPs
2   for k = 1, ..., K do                                      # outer NPL loop
3       F_pi(s, s') := sum_a pi-hat(a | s) * P_a(s, s')      # policy-weighted transitions
4       e_pi(s, a) := gamma - log pi-hat(a | s)               # emax correction
5       W_phi, W_e := (I - beta * F_pi)^{-1} applied to      # Hotz-Miller inversion
              (sum_a pi-hat * phi, sum_a pi-hat * e_pi)
6       z-tilde(s, a) := phi(s, a) + beta * sum_{s'} P_a(s, s') W_phi(s')
7       e-tilde(s, a) := beta * sum_{s'} P_a(s, s') W_e(s')
8       theta_k := argmax_theta sum_{i,t} log-softmax(        # L-BFGS-B step
              (z-tilde(s_it, *)' theta + e-tilde(s_it, *)) / sigma)[a_it]
9       pi-hat(a | s) := softmax(                             # update policy
              (z-tilde(s, *)' theta_k + e-tilde(s, *)) / sigma)[a]
10      if ||theta_k - theta_{k-1}|| < tol: break             # NPL convergence
11  return theta_hat = theta_K, standard errors, pi-hat, V
```

The outer optimizer in step 8 is L-BFGS-B, applied to the augmented-feature logit;
the gradient is the closed-form score $\psi_i$ from the Estimator section. The
default in the public `CCP` wrapper is `num_policy_iterations=1`: the loop runs
once, giving the one-step Hotz-Miller estimator, which is consistent but
asymptotically less efficient than full MLE. Setting `num_policy_iterations=K` for
$K > 1$ runs the NPL iteration for $K$ steps; setting $K = -1$ continues until
parameter convergence. The implementation lives in `econirl.estimation.ccp`.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| States and actions are discrete. | Many states have weak or one-action empirical support. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling challenge. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| Rapid structural estimates are needed. | The reference nested fixed-point likelihood is required. |
| Empirical action support is strong across all states. | First-stage choice probabilities are sparse or imputed. |

CCP targets the same structural reward object as NFXP in finite tabular dynamic
discrete choice models and is preferred when the cost of repeated Bellman solves
dominates. NFXP remains the direct maximum-likelihood reference and is preferred when
first-stage policy support is a concern. MPEC reformulates the same MLE problem as a
constrained program. NNES and TD-CCP become attractive when exact matrix inversion
or exact Bellman solves are too costly. Behavioral cloning is a weaker baseline
because it stops at the first-stage policy without recovering structural rewards.

## Usage

```python
from econirl.datasets import load_rust_bus
from econirl import CCP

df = load_rust_bus()

model = CCP(
    n_states=90,
    discount=0.9999,
    utility="linear_cost",
    num_policy_iterations=10,   # NPL; set 1 for one-step Hotz-Miller
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

Counterfactual analysis re-solves the fitted dynamic program under a changed
primitive:

```python
cf = model.counterfactual(RC=4.0)      # raise the replacement cost
print(cf.params)
print(cf.policy)                        # new replacement probability by state
```

The fitted policy gives the replacement probability by state, which can be read at
specific states:

```python
print(model.predict_proba([0, 10, 50, 89]))
```

The [Quick Start](ccp/quick_start.md) page documents the full set of fitted
attributes and the lower-level `CCPEstimator` interface.

## Evidence

Parameter recovery is measured on the `canonical_low_action` synthetic cell, which
has known rewards, transitions, policies, values, Q functions, and Type A, Type B,
and Type C counterfactual oracles. The figure below is a Monte-Carlo study over 200
replications: the panel is resimulated and refit on a fresh seed each time, and each
parameter is plotted as its recovered mean and 95% interval against the true value.

![ccp parameter recovery, Monte Carlo](../_static/estimators/ccp_recovery.png)

| Parameter | True | Recovered (mean) | 95% interval |
| --- | ---: | ---: | --- |
| `action_0_intercept` | 0.10 | _pending MC_ | _pending MC_ |
| `action_0_progress` | 0.50 | _pending MC_ | _pending MC_ |
| `action_1_intercept` | 0.00 | _pending MC_ | _pending MC_ |
| `action_1_progress` | -0.20 | _pending MC_ | _pending MC_ |

Recovery numbers are over a 200-replication Monte-Carlo run; numbers pending the
full run.

Behavioral fit and counterfactual regret on the same cell, against the known oracle
objects:

| Metric | Value |
| --- | ---: |
| Policy total variation | 0.0057 |
| Value RMSE | 0.0194 |
| Type A regret (reward shift) | 0.000213 |
| Type B regret (transition change) | 0.000362 |
| Type C regret (action removed) | 0.000086 |

The regrets are small because the recovered reward is close enough to the truth that
re-solving the intervened model reproduces almost the same policy as the oracle.
For the full cross-estimator comparison, see the
[bus engine simulation study](../simulation_studies/rust_bus.md).

## References

Source papers:

- Hotz, V. J., and Miller, R. A. (1993). "Conditional Choice Probabilities and the
  Estimation of Dynamic Models." _Review of Economic Studies_, 60(3), 497-529.
  {ref}`reference entry <hotz-miller-1993>`.
- Aguirregabiria, V., and Mira, P. (2002). "Swapping the Nested Fixed Point
  Algorithm: A Class of Estimators for Discrete Markov Decision Models."
  _Econometrica_, 70(4), 1519-1543.
  {ref}`reference entry <aguirregabiria-mira-2002>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.ccp`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/ccp.py).
- sklearn wrapper: [`econirl.CCP`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/ccp.py).
- Validation runner: [`validation/estimators/ccp/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ccp/run.py).
- Recovery study: [`validation/estimators/ccp/recovery_mc.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ccp/recovery_mc.py).
- Results files: [`ccp.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp.json).

Pages:

- [Quick Start](ccp/quick_start.md)
- [Pre-Estimation Checks](ccp/pre_estimation.md)
- [Simulation Study](ccp/validation.md)
- [Counterfactuals](ccp/counterfactuals.md)
- [Rust Bus Engine Example](ccp/rust_bus.md)

```{toctree}
:hidden:

ccp/quick_start
ccp/pre_estimation
ccp/validation
ccp/counterfactuals
ccp/rust_bus
```
