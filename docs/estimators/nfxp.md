# NFXP

## Important Links

- [Quick Start](nfxp/quick_start.md)
- [Pre-Estimation Checks](nfxp/pre_estimation.md)
- [Simulation Study](nfxp/validation.md)
- [Counterfactuals](nfxp/counterfactuals.md)
- [Bus Engine Example](nfxp/rust_bus.md)

Nested fixed point is the reference estimator for tabular structural dynamic
discrete choice. It recovers primitive reward parameters by maximum likelihood,
nesting the solution of the agent's dynamic program inside each likelihood
evaluation. The inner loop solves the Bellman fixed point for a candidate
reward parameter. The outer loop maximizes the conditional choice log
likelihood over that parameter.

NFXP is the benchmark for later structural estimators. These estimators either
retain its target with a different numerical method or relax one of its
bottlenecks. Each estimator's page describes the resulting tradeoff.

## Source Papers

The estimator follows {ref}`Rust (1987) <rust-1987>`, which introduces the
bus-engine replacement model and the nested fixed-point algorithm.
{ref}`Iskhakov et al. (2016) <iskhakov-2016>` compare the nested fixed point to
constrained-optimization alternatives and motivate the polyalgorithm inner
solver used in the package.

## Notation

The discrete state is $s$ and the action is $a$. Individual $i$ is observed in
period $t$. The index $b$ denotes an action inside a sum. The vector
$\phi(s, a)$ contains the known reward features. The parameter vector is
$\theta$, the discount factor is $\beta$, and the logit shock scale is $\sigma$.

The transition kernel $P_a(s, s')$ gives the probability of moving from $s$ to
$s'$ under action $a$. It is stored in $(A, S, S)$ orientation. The integrated
value is $V_\theta(s)$. The choice-specific value is $Q_\theta(s, a)$. The
policy is $\pi_\theta(a \mid s)$.

## Model

The observed data are state, action, and next-state trajectories
$(s_{it}, a_{it}, s_{i,t+1})$. The flow payoff is linear in the features:

$$
u_\theta(s, a) = \phi(s, a)^\top \theta.
$$

With discount factor $\beta$ and logit shock scale $\sigma$, the integrated value
function solves the soft Bellman fixed point $V_\theta = T_\theta V_\theta$,
where the Bellman operator $T_\theta$ is defined by:

$$
V_\theta(s)
= \sigma \log \sum_a \exp\!\left(
    \frac{u_\theta(s, a) + \beta \sum_{s'} P_a(s, s') V_\theta(s')}{\sigma}
\right).
$$

The choice-specific value is:

$$
Q_\theta(s, a) = u_\theta(s, a) + \beta \sum_{s'} P_a(s, s') V_\theta(s').
$$

The implied conditional choice probability follows the logit rule:

$$
\pi_\theta(a \mid s) =
\frac{\exp(Q_\theta(s, a) / \sigma)}
     {\sum_b \exp(Q_\theta(s, b) / \sigma)}.
$$

The canonical instance is Rust's bus-engine replacement model. A bus operator
decides each period whether to keep a deteriorating engine or pay a flat cost to
replace it. The dynamic program links today's choices to tomorrow's states, so
observed choices carry information about the structural costs.

## Identification

This section states when the estimated parameters can be interpreted as
primitive reward parameters rather than only as an in-sample choice fit.

Interpreting $\theta$ as structural reward parameters requires the following
assumptions.

- **Conditional independence (CI).** The observed state transition is Markov in
  the current state and action and does not depend on the current logit shock.
- **Additive separability (AS).** The per-period payoff is the systematic reward
  plus an additive choice-specific shock, drawn independently across choices as
  Type-I extreme value with fixed scale $\sigma$.
- **Exogenous transitions.** The transition kernel $P_a(s, s')$ is supplied or
  estimated from observed transitions. The default standard errors condition
  on this estimate. Full-likelihood BHHH also incorporates transition-score
  uncertainty.
- **Reward normalization.** The Type-I extreme-value shock scale is fixed at 1.
  The supplied reward specification must exclude reward components that are
  common to every action. NFXP rejects rank-deficient action contrasts.
- **Action-dependent feature rank.** The reward features must vary across
  actions. The feature rank must equal the number of parameters. State-only
  features copied across actions collapse the action contrasts and leave $\theta$
  unidentified.

These assumptions apply within a finite discrete state space and a stationary
environment. The model also assumes expected-utility maximization and a known
discount factor $\beta$. Thin action support, an invalid normalization, or an
incorrectly oriented transition tensor can make the estimates hard to
interpret.

## Estimator

NFXP maximizes the conditional log likelihood:

$$
\hat{\theta}
= \arg\max_\theta \sum_{i,t} \log \pi_\theta(a_{it} \mid s_{it}).
$$

Because $V_\theta = T_\theta V_\theta$ is an implicit equation in $\theta$, the
derivative $\partial V/\partial\theta$ is obtained by the implicit function
theorem rather than by differentiating through the iteration.

The score follows from differentiating $\log \pi_\theta(a \mid s)$ directly.
Writing out the softmax log:

$$
\frac{\partial}{\partial\theta} \log \pi_\theta(a \mid s)
= \frac{\partial}{\partial\theta}
  \!\left[
    \frac{Q_\theta(s,a)}{\sigma}
    - \log \sum_b \exp\!\left(\frac{Q_\theta(s,b)}{\sigma}\right)
  \right].
$$

Apply the chain rule. The log-sum-exp derivative is
$\sum_b \pi_b\,\partial f_b/\partial\theta$:

$$
= \frac{1}{\sigma}\frac{\partial Q_\theta(s,a)}{\partial\theta}
  - \frac{1}{\sigma}\sum_b \pi_\theta(b \mid s)
    \frac{\partial Q_\theta(s,b)}{\partial\theta}.
$$

The per-observation score at observation $i$ is:

$$
\psi_i(\theta)
=
\frac{1}{\sigma}
\left[
    \frac{\partial Q_\theta(s_i, a_i)}{\partial \theta}
    -
    \sum_b \pi_\theta(b \mid s_i)
    \frac{\partial Q_\theta(s_i, b)}{\partial \theta}
\right].
$$

The MLE first-order condition is $\sum_{i,t} \psi_i(\theta) = 0$. BHHH updates
the parameters iteratively to solve this condition.

The Q-gradient follows from the chain rule on
$Q_\theta(s,a) = u_\theta(s,a) + \beta\sum_{s'} P_a(s,s') V_\theta(s')$:

$$
\frac{\partial Q_\theta(s,a)}{\partial\theta}
= \phi(s,a)
  + \beta \sum_{s'} P_a(s,s')\frac{\partial V_\theta(s')}{\partial\theta}.
$$

Differentiate the soft Bellman value with the log-sum-exp derivative:

$$
\frac{\partial V_\theta(s)}{\partial\theta}
= \sum_a \pi_\theta(a \mid s)
  \frac{\partial Q_\theta(s,a)}{\partial\theta}.
$$

Substituting the Q-gradient into this value derivative and collecting
$\partial V/\partial\theta$ terms on the left yields the linear system:

$$
(I - \beta P_\pi)\frac{\partial V}{\partial \theta}
= \sum_a \pi_\theta(a \mid s)\,\phi(s, a),
$$

where $P_\pi = \sum_a \operatorname{diag}_s(\pi_\theta(a \mid s))\, P_a \in \mathbb{R}^{S \times S}$
is the policy-weighted transition matrix, with $\operatorname{diag}_s(\pi_\theta(a \mid s))$
denoting the $S \times S$ diagonal matrix whose $(s,s)$ entry is $\pi_\theta(a \mid s)$.

## System View

NFXP is easiest to read as two nested questions. The outside question asks which
reward parameters make the observed choices most likely. The inside question
asks how a forward-looking agent would behave if those parameters were true.

```text
Observed panel: state, action, next state
Reward features, transition model, discount factor
        |
        v
Try one candidate reward parameter theta
        |
        v
Solve the agent's dynamic program
        |
        v
Convert values into logit choice probabilities
        |
        v
Score the observed actions under those probabilities
        |
        v
Update theta and repeat until the likelihood is maximized
```

Use NFXP when that inside solve is affordable. The estimated reward, value
function, policy, and counterfactuals all come from the same fully specified
dynamic choice model.

## Algorithm

```text
Algorithm  NFXP (nested fixed-point maximum likelihood)
Input   panel {(s_it, a_it, s_{i,t+1})}, features phi, transitions P,
        discount beta, logit scale sigma
Output  theta_hat, standard errors, policy pi, value V

1   initialize theta
2   repeat                                         # outer loop: BHHH ascent
3       u_theta(s, a) := phi(s, a)' theta
4       solve  V_theta = T_theta V_theta           # inner loop: Bellman fixed point
5       Q_theta(s, a) := u_theta(s, a) + beta * sum_{s'} P_a(s, s') V_theta(s')
6       pi_theta(a | s) := exp(Q_theta(s, a)/sigma) / sum_b exp(Q_theta(s, b)/sigma)
7       L(theta) := sum_{i,t} log pi_theta(a_it | s_it)
8a      H <- sum_i psi_i(theta) psi_i(theta)^T   # BHHH information matrix
8b      theta <- theta + H^{-1} (sum_i psi_i(theta))  # Newton-like ascent step
9   until g(theta)' H(theta)^{-1} g(theta) is below tolerance
10  return theta_hat, standard errors from the selected covariance method, pi_theta, V_theta
```

The inner solve in step 4 defaults to `inner_solver="polyalgorithm"`. Following
Iskhakov et al. (2016), it uses successive approximation far from the fixed
point and Newton-Kantorovich steps near the solution. The pure `sa` and `nk`
variants are also available. Successive approximation is linearly convergent
and robust from any start. Newton-Kantorovich is quadratically convergent near
the solution and needs a good starting point. The outer optimizer is BHHH. It
approximates the information matrix with outer products of the per-observation
scores.

The standard errors come from maximum-likelihood asymptotics. The `se_method`
argument selects the covariance estimator. `asymptotic` inverts the
observed-information matrix. `robust`, the default, returns the sandwich
covariance. It places the outer product of the per-observation scores between
two copies of the inverse information matrix. `clustered` sums scores by individual
before forming the middle matrix. `bootstrap` resamples whole trajectories. For
the Rust Table IX replication, `full_likelihood_bhhh` forms the BHHH outer
product for the joint structural and transition-probability likelihood and
reports the structural covariance block. Under correct specification, the
`asymptotic` and `robust` covariance estimators have the same large-sample
limit. They can differ under misspecification.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| States and actions are discrete. | The state space is too large for repeated Bellman solves. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling challenge. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| A structural reference estimate is required. | Only a fast imitation baseline is required. |
| Counterfactual policy analysis is central. | Only fitted choice probabilities are required. |

## Usage

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NFXP

df = load_rust_bus()

model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

params = {name: round(value, 6) for name, value in model.params_.items()}
print(params)
print(f"P(replace | state=50) = {model.predict_proba([50])[0, 1]:.6f}")
```

**Result**

```text
{'operating_cost': 0.001003, 'replacement_cost': 3.072264}
P(replace | state=50) = 0.086333
```

The [Simulation Study](nfxp/validation.md) shows the complete `summary()`
report from a 200-state fit.

Counterfactual analysis re-solves the fitted dynamic program after changing a
model parameter:

```python
replacement_cost = 4.0
cf = model.counterfactual(replacement_cost=replacement_cost)
print(
    f"replacement_cost: {model.params_['replacement_cost']:.6f}"
    f" -> {replacement_cost:.6f}"
)
print(
    f"P(replace | state=50): {model.predict_proba([50])[0, 1]:.6f}"
    f" -> {cf.counterfactual_policy[50, 1]:.6f}"
)
```

**Result**

```text
replacement_cost: 3.072264 -> 4.000000
P(replace | state=50): 0.086333 -> 0.055197
```

Transition counterfactuals use a complete action-specific transition model.
The [Counterfactuals](nfxp/counterfactuals.md) page gives a runnable example
and its result.

See [Quick Start](nfxp/quick_start.md) for the advanced `NFXPEstimator` API.

## Evidence

Three experiments assess NFXP estimation, inference, and counterfactual
performance.

- **Estimation.** Twenty independent panels have 200 states, two actions, three
  reward parameters, and 7,500 observations each. The mean policy
  total-variation distance from the true policy is 0.0085.
- **Inference.** In one thousand independent 40-state panels, coverage of the
  three nominal 95 percent intervals ranges from 94.8 to 95.4 percent.
- **Counterfactuals.** The interventions change the reward and slow engine
  deterioration. The mean policy total-variation distances from the
  corresponding true-parameter policies are 0.0064 and 0.0067.

The [Simulation Study](nfxp/validation.md) reports the estimation, inference,
and counterfactual results in detail. The
[bus engine simulation study](../simulation_studies/rust_bus.md) compares NFXP
with other estimators on a shared problem.

## References

### Source Papers

- Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of
  Harold Zurcher. _Econometrica_, 55(5), 999-1033.
  {ref}`reference entry <rust-1987>`.
- Iskhakov, F., Lee, J., Rust, J., Schjerning, B., and Seo, K. (2016). Comment on
  "Constrained Optimization Approaches to Estimation of Structural Models."
  _Econometrica_, 84(1), 365-370. {ref}`reference entry <iskhakov-2016>`.

### Code and Results

- Estimator source: [`econirl.estimation.nfxp`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/nfxp.py).
- sklearn wrapper: [`econirl.NFXP`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/nfxp.py).
- Simulation code: [`validation/estimators/nfxp/ready.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nfxp/ready.py).
- Simulation results: [`validation/results/nfxp_ready.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp_ready.json).

### Pages

- [Quick Start](nfxp/quick_start.md)
- [Pre-Estimation Checks](nfxp/pre_estimation.md)
- [Simulation Study](nfxp/validation.md)
- [Counterfactuals](nfxp/counterfactuals.md)
- [Rust Bus Engine Example](nfxp/rust_bus.md)

```{toctree}
:hidden:

nfxp/quick_start
nfxp/pre_estimation
nfxp/validation
nfxp/counterfactuals
nfxp/rust_bus
```
