# CCP

## Important Links

- [Quick Start](ccp/quick_start.md)
- [Pre-Estimation Checks](ccp/pre_estimation.md)
- [Simulation Study](ccp/validation.md)
- [Counterfactuals](ccp/counterfactuals.md)
- [Bus Engine Example](ccp/rust_bus.md)

Conditional choice probability estimation recovers primitive reward parameters
from tabular dynamic discrete choice data. It replaces repeated Bellman
fixed-point computations with a matrix inversion at each NPL step. The first
stage estimates conditional choice probabilities from state-action
frequencies. The Hotz-Miller inversion maps those probabilities to normalized
differences in conditional values. A policy valuation step then combines the
choice probabilities with rewards, shocks, and transitions. The
Aguirregabiria-Mira nested pseudo-likelihood extension alternates policy
valuation with pseudo-likelihood optimization.

CCP and NFXP target the same finite structural model. CCP requires enough
first-stage action support for the inversion.

## Source Papers

The estimator follows {ref}`Hotz and Miller (1993) <hotz-miller-1993>`, which
introduces the inversion theorem and its two-stage estimator.
{ref}`Aguirregabiria and Mira (2002) <aguirregabiria-mira-2002>` introduces
nested pseudo-likelihood. Under their assumptions, every fixed K-stage
estimator is first-order equivalent to the partial maximum likelihood
estimator. Additional stages can improve finite-sample precision.
{ref}`Magnac and Thesmar (2002)
<magnac-thesmar-2002>` gives the identification boundary for dynamic
primitives.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action, observed for
individual $i$ in period $t$. The vector $\phi(s, a)$ collects the known reward
features and $\theta$ the reward parameters to be estimated. The discount factor is
$\beta$ and the logit shock scale is $\sigma$. The additive choice-specific
shock is $\varepsilon_a$. It is drawn independently across choices as Type-I
extreme value with scale $\sigma$. Its conditional expectation given optimality
is $\sigma[\gamma - \log \hat\pi(a \mid s)]$ in utility units. We write
$e_{\hat\pi}(s,a)=\gamma-\log\hat\pi(a\mid s)$ for the normalized correction.
The transition kernel $P_a(s, s')$
gives the probability of moving to $s'$ from $s$ under action $a$, stored in
$(A, S, S)$ orientation. The first-stage empirical conditional choice probability is
$\hat\pi(a \mid s)$, estimated from state-action frequencies. The Euler-Mascheroni
constant $\gamma \approx 0.5772$ enters the logit emax correction. The
policy-weighted transition matrix $F_{\hat\pi}$, augmented feature weights
$W_\phi(s)$ and $W_e(s)$, and the pseudo choice-specific value
$\tilde{Q}_\theta(s, a;\hat\pi)$ are defined in the Model section.

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

The correction $e_{\hat\pi}$ is dimensionless. In the theoretical
representation, it enters utility as $\sigma e_{\hat\pi}$. The Euler
constant contributes the same value offset, $\sigma\gamma/(1-\beta)$, to every
state. The implementation removes that offset before policy valuation and
therefore uses $-\log\hat\pi(a\mid s)$. This normalization leaves choice
probabilities unchanged. They are computed from $\exp(\tilde Q/\sigma)$.

The integrated Bellman equation under $\hat\pi$ is:

$$
\bar{V}_{\hat\pi}(s)
= \sum_a \hat\pi(a \mid s)
  \bigl[u_\theta(s, a) + \sigma e_{\hat\pi}(s, a)
        + \beta \sum_{s'} P_a(s, s')\, \bar{V}_{\hat\pi}(s')\bigr].
$$

Collecting the $\bar{V}$ terms, this becomes
$(I - \beta F_{\hat\pi})\bar{V}_{\hat\pi} = \sum_a \hat\pi(a \mid s)\{u_\theta(s, a) + \sigma e_{\hat\pi}(s, a)\}$,
which inverts to:

$$
\bar{V}_{\hat\pi}
= (I - \beta F_{\hat\pi})^{-1}
  \sum_a \hat\pi(a \mid s)
  \bigl\{u_\theta(s, a) + \sigma e_{\hat\pi}(s, a)\bigr\}.
$$

For linear rewards, this separates into parameter-dependent and parameter-free parts:

$$
\bar{V}_{\hat\pi}(s) = W_\phi(s)^\top \theta + W_e(s),
$$

where:

$$
W_\phi = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\, \phi(s, a),
\qquad
W_e = (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\, \sigma e_{\hat\pi}(s, a).
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
$\theta$. One inversion of $(I - \beta F_{\hat\pi})$ per NPL step replaces the
repeated Bellman solves that NFXP performs during likelihood evaluation.

**Gradient with respect to $\theta$.**
With $\hat\pi$ fixed, differentiate the fixed-policy Bellman equation
$(I - \beta F_{\hat\pi})\bar{V}_{\hat\pi} = \sum_a \hat\pi(a \mid s)\,[\phi(s,a)^\top\theta + \sigma e_{\hat\pi}(s,a)]$
to obtain the sensitivity of $\bar{V}_{\hat\pi}$ to $\theta$:

$$
(I - \beta F_{\hat\pi})\frac{\partial \bar{V}_{\hat\pi}}{\partial\theta}
= \sum_a \hat\pi(a \mid s)\,\phi(s, a).
$$

Because $\hat\pi$ and $F_{\hat\pi}$ do not depend on $\theta$, the left-hand
side is linear in $\partial \bar{V}/\partial\theta$ and the right-hand side is
constant.  The unique solution is:

$$
W_\phi \;=\; \frac{\partial \bar{V}_{\hat\pi}}{\partial\theta}
\;=\; (I - \beta F_{\hat\pi})^{-1} \sum_a \hat\pi(a \mid s)\,\phi(s, a).
$$

This is the derivative of the CCP-implied value function with respect to
$\theta$. It requires one linear solve. The policy is fixed during each NPL
inner step, so the derivative does not propagate through the outer policy
update. See Aguirregabiria and Mira (2002) eqs. (4)-(5).

## Identification

CCP identifies $\theta$ only within the stated parametric model and
normalization. The following restrictions define that interpretation.

- **Conditional Independence (CI).** The observed state transition is Markov in the
  current state and action and does not depend on the current logit shock.
- **Additive Separability (AS).** The per-period payoff is the systematic reward plus
  an additive choice-specific shock drawn independently across choices as Type-I
  extreme value with fixed scale $\sigma$.
- **Transition Law.** The transition kernel $P_a(s, s')$ is supplied or
  estimated before the payoff fit. The public wrapper does not estimate
  transition uncertainty jointly with reward parameters.
- **Fixed Dynamic Primitives.** The discount factor, Type-I extreme-value shock
  family, and shock scale are fixed.
- **Reward Normalization.** A reference payoff or another explicit restriction
  fixes the reward location. Fixing $\sigma$ fixes the utility scale.
- **Dynamic Feature Rank.** Reward parameters must change observed choice
  values through current payoff contrasts or action-dependent future state
  distributions. The public wrapper requires full direct action-contrast rank
  and stops when this check fails. A richer identification analysis may
  recover state-only features through action-dependent transitions, but that
  case is outside the wrapper's accepted scope.

These are population restrictions. Empirical action support and state coverage
are separate finite-sample requirements. Near-zero estimated CCPs make
$\gamma-\log\hat\pi(a\mid s)$ unstable. Sparse states make the first-stage
policy noisy. Smoothing prevents numerical zeros, but it does not add
identifying information.

The implemented scope is finite-state, stationary, linear-utility logit CCP.
Continuous states, serially correlated private shocks, nonlogit shock
distributions, and latent finite types require other estimators or additional
machinery.

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
  \right],
$$

where $(s_i, a_i)$ stands for a single observation $(s_{it}, a_{it})$ from the panel.

The pseudo-MLE first-order condition is:

$$
\sum_{i,t} \psi_i(\hat\theta_k) = 0,
$$

which the inner L-BFGS-B step solves numerically.

The score $\psi_i$ drives the inner L-BFGS-B step. The asymptotic,
robust, and clustered standard errors use the Hessian and observation scores
from the same fixed-CCP pseudo-likelihood. They condition on the policy object
and the supplied transition model. They are not the full Hotz-Miller two-step
covariance. For a converged NPL fit with transitions estimated from the fitted
panel, `se_method="full_likelihood_bhhh"` instead uses the joint
reward-transition likelihood. It is unavailable for fixed-stage fits or
supplied transitions.

Pairs-cluster bootstrap standard errors resample individuals and repeat the
empirical CCP and reward fit. The supplied transition tensor remains fixed.
This is not the parametric one-step NPL bootstrap of Kasahara and Shimotsu
(2008). Clustered inference also relies on independent clusters and enough
clusters for the across-cluster approximation.

## System View

CCP keeps the structural target from NFXP but changes where the computation
happens. Instead of solving a new dynamic program for every trial parameter, it
first reads the policy from observed choice frequencies and uses that policy to
build a one-shot continuation-value correction.

```text
Observed panel: state, action, next state
Reward features, transition model, discount factor
        |
        v
Estimate first-stage choice probabilities by state
        |
        v
Invert the policy into continuation-value terms
        |
        v
Fit a logit pseudo-likelihood for theta
        |
        +---- stop after one Hotz-Miller step
        |
        +---- or update the policy and iterate as NPL
        |
        v
Recovered reward parameters and implied policy
```

The tradeoff is support. CCP is fast when every relevant state-action cell
is observed often enough. Thin cells make the first-stage policy noisy, and the
inversion turns that noise into reward noise.

## Algorithm

```text
Algorithm  CCP/NPL (conditional choice probability, K-step nested pseudo-likelihood)
Input   panel {(s_it, a_it, s_{i,t+1})}, features phi, transitions P,
        discount beta, logit scale sigma, NPL steps K
Output  theta_hat, standard errors, policy pi^K, value V

1   estimate pi^0(a | s) from state-action frequencies        # first-stage CCPs
2   for k = 1, ..., K do                                      # outer NPL loop
3       F_pi(s, s') := sum_a pi^{k-1}(a | s) * P_a(s, s')   # policy-weighted transitions
4       e_pi(s, a) := -log pi^{k-1}(a | s)                   # value-level normalization
5       W_phi, W_e := (I - beta * F_pi)^{-1} applied to      # Hotz-Miller inversion
              (sum_a pi^{k-1} * phi, sum_a pi^{k-1} * sigma * e_pi)
6       z-tilde(s, a) := phi(s, a) + beta * sum_{s'} P_a(s, s') W_phi(s')
7       e-tilde(s, a) := beta * sum_{s'} P_a(s, s') W_e(s')
8       theta_k := argmax_theta sum_{i,t} log-softmax(        # L-BFGS-B step
              (z-tilde(s_it, *)' theta + e-tilde(s_it, *)) / sigma)[a_it]
9       pi^k(a | s) := softmax(                               # update policy
              (z-tilde(s, *)' theta_k + e-tilde(s, *)) / sigma)[a]
10      r_theta := ||theta_k - theta_{k-1}||_2
11      r_policy := max |pi^k - pi^{k-1}|
12      if r_theta <= tol and r_policy <= tol: break          # NPL fixed point
13  return theta_hat = theta_K, standard errors, pi^K, V
```

The inner optimizer in step 8 is L-BFGS-B. It fits the augmented-feature logit.
Its gradient is the closed-form score $\psi_i$ from the Estimator section. The
public `CCP` wrapper defaults to `num_policy_iterations=1`. The loop runs once
and returns the one-step Hotz-Miller estimator. Under the assumptions in
Aguirregabiria and Mira (2002), every fixed $K\geq1$ estimator is consistent,
asymptotically normal, and first-order equivalent to partial maximum
likelihood. Their simulations find the largest finite-sample gain between the
first and second stages.

For $K>1$, `num_policy_iterations=K` is a maximum. The run can stop early only
when both residuals in steps 10 and 11 meet the tolerance. Reaching a positive
stage maximum without that fixed point is a successful `fixed_k_complete`
run. Setting $K=-1$ requires the joint fixed point. Reaching the safety cap in
that mode is a failed `iteration_cap_reached` run.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| States and actions are discrete. | Many states have weak or one-action empirical support. |
| Transitions are known or can be estimated first. | Transition estimation is the main modeling challenge. |
| The reward has a compact parametric form. | The reward must be high-dimensional or neural. |
| Rapid structural estimates are needed. | The reference nested fixed-point likelihood is required. |
| Empirical action support is strong across all states. | First-stage choice probabilities are sparse or imputed. |

CCP and NFXP target the same structural reward in finite, tabular dynamic
discrete choice models. CCP avoids repeated Bellman solves. NFXP remains the
direct maximum-likelihood reference. It is preferable when first-stage policy
support is weak. MPEC reformulates the same MLE problem as a constrained
program. NNES and TD-CCP are alternatives when exact matrix inversion or
Bellman solves cost too much. Behavioral cloning stops at the first-stage
policy and does not recover structural rewards.

## Usage

```python
from econirl import CCP
from econirl.datasets import load_rust_bus, rust_bus_reward_spec

df = load_rust_bus()

model = CCP(
    n_states=90,
    discount=0.9999,
    utility=rust_bus_reward_spec(90),
    num_policy_iterations=3,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

for name in model.params_:
    print(f"{name}: estimate={model.params_[name]:.6f}, se={model.se_[name]:.6f}")
print(f"termination={model.termination_reason_}")
print(f"npl_converged={model.npl_converged_}")
print(f"parameter_residual={model.npl_parameter_residual_:.6e}")
print(f"policy_residual={model.npl_policy_residual_:.6e}")
```

**Result**

```text
operating_cost: estimate=0.000995, se=0.000421
replacement_cost: estimate=3.072211, se=0.074237
termination=fixed_k_complete
npl_converged=False
parameter_residual=1.452655e-02
policy_residual=2.148578e-02
```

A fixed three-step NPL run completed the requested work. The separate
`npl_converged_` value says whether both the parameter and policy residuals met
the fixed-point tolerance. The residuals show how far the final stage is from
that joint stopping rule.

Counterfactual analysis re-solves the fitted dynamic program under a changed
primitive:

```python
cf = model.counterfactual(replacement_cost=4.0)
print(f"replacement_cost={cf.params['replacement_cost']:.6f}")
print(f"P(replace | state=50)={cf.policy[50, 1]:.6f}")
```

**Result**

```text
replacement_cost=4.000000
P(replace | state=50)=0.054908
```

The fitted policy reports the replacement probability for each state:

```python
for state, row in zip(
    [0, 10, 50, 89],
    model.predict_proba([0, 10, 50, 89]),
    strict=True,
):
    print(f"{state}: keep={row[0]:.6f}, replace={row[1]:.6f}")
```

**Result**

```text
0: keep=0.955732, replace=0.044268
10: keep=0.947600, replace=0.052400
50: keep=0.912779, replace=0.087221
89: keep=0.884699, replace=0.115301
```

See [Quick Start](ccp/quick_start.md) for the complete `CCPEstimator` API.

## Evidence

On the official STORDAT Group-4 panel, converged NPL reproduces the published
Rust Table IX point estimates, standard errors, transition probabilities,
choice log likelihood, and full log likelihood to the reported precision. It
also agrees with the reference NFXP fit.

The inference experiment contains 1,000 independent 20-state panels. Every
panel produced a finite estimate and robust standard errors. The study supplies
the true transition tensor. It evaluates reward estimation and conditional
pseudo-likelihood inference, not uncertainty from transition estimation.

| Parameter | True value | Mean estimate | Empirical SD | Mean SE | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reward feature 1 | 0.35 | 0.354 | 0.0305 | 0.0296 | 94.2% |
| Reward feature 2 | -0.25 | -0.248 | 0.0506 | 0.0499 | 95.4% |
| Reward feature 3 | 0.20 | 0.203 | 0.0521 | 0.0543 | 96.3% |

The larger recovery experiment uses three-stage NPL on 20 independent
100-state panels. Median relative parameter errors range from 1.9 to 5.2
percent. The mean policy distance is 0.0028.

Reward and transition counterfactuals have mean policy TV below 0.0023. Their
mean value losses are below 0.0002. The reported comparison also applies
four conditional inference procedures to one panel. The transition tensor is
held fixed in all four.

See the [Simulation Study](ccp/validation.md) for the full design, results, and
reproduction command. For the cross-estimator comparison, see the
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
- Magnac, T., and Thesmar, D. (2002). "Identifying Dynamic Discrete Decision
  Processes." _Econometrica_, 70(2), 801-816.
  {ref}`reference entry <magnac-thesmar-2002>`.
- Aguirregabiria, V., and Mira, P. (2010). "Dynamic Discrete Choice Structural
  Models: A Survey." _Journal of Econometrics_, 156(1), 38-67.
  {ref}`reference entry <aguirregabiria-mira-2010>`.
- Kasahara, H., and Shimotsu, K. (2008). "Pseudo-Likelihood Estimation and
  Bootstrap Inference for Structural Discrete Markov Decision Models."
  _Journal of Econometrics_, 146(1), 92-106.
  {ref}`reference entry <kasahara-shimotsu-2008>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.ccp`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/ccp.py).
- sklearn wrapper: [`econirl.CCP`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/ccp.py).
- Simulation runner: [`validation/estimators/ccp/ready.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ccp/ready.py).
- Results file: [`ccp_ready.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp_ready.json).

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
