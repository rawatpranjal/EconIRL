# Under the Hood

SEES estimates a structural dynamic discrete choice likelihood while replacing
the full Bellman solution object with a sieve approximation. The default path
is V-SEES, which approximates the value function.

## Optimization Setup

The observed panel supplies state, action, and next-state records. The
transition law is estimated from the panel or supplied directly. Reward
features, basis functions, the discount factor, the logit shock scale, the
Bellman-residual penalty weight, and the solution mode are fixed before
optimization.

SEES optimizes the structural reward parameters `theta` and sieve
coefficients jointly. The objective is a penalized conditional log likelihood:
the log likelihood rewards fit to observed actions, while the Bellman residual
penalty keeps the approximated dynamic object close to a Bellman-consistent
solution. The public simulation path uses the `value` mode unless another
mode is explicitly selected for diagnostics.

## Model

The observed data are state, action, and next-state trajectories.

$$
(s_{it}, a_{it}, s_{i,t+1})
$$

The flow payoff is finite-dimensional.

$$
u_\theta(s, a) = \phi(s, a)^\top \theta
$$

For V-SEES, the value function is approximated by basis functions.

$$
V_\alpha(s) = \Psi(s)^\top \alpha
$$

The choice-specific value is

$$
Q_{\theta,\alpha}(s,a)
= u_\theta(s,a)
  + \beta \sum_{s'} P_a(s,s') V_\alpha(s').
$$

Choice probabilities follow the soft-max rule.

$$
\pi_{\theta,\alpha}(a \mid s)
=
\frac{\exp(Q_{\theta,\alpha}(s,a) / \sigma)}
     {\sum_b \exp(Q_{\theta,\alpha}(s,b) / \sigma)}.
$$

SEES maximizes the log likelihood with a Bellman-residual penalty.

$$
(\hat{\theta}, \hat{\alpha})
= \arg\max_{\theta,\alpha}
    \ell(\theta,\alpha)
    - \omega \lVert V_\alpha - T_\theta(V_\alpha) \rVert^2.
$$

The penalty weight controls how strongly the estimated value approximation
must satisfy the Bellman equation.

For nonconvex finite-sample problems, SEES can run deterministic theta
multistart before the joint theta-alpha optimization. Set
`num_theta_starts > 1` to include the supplied start, a static-logit start,
and neutral variants; the selected fit is the one with the best penalized
criterion.

## Pseudocode

```text
Input: panel, reward features, transitions, basis functions, and penalty weight
Initialize theta and sieve coefficients
Optionally build deterministic theta starts
for each start:
    compute the approximated value or Q object from the basis
    compute choice probabilities and the log likelihood
    compute the Bellman residual penalty
    optimize theta and sieve coefficients jointly
Select the best penalized objective
return theta, sieve object, policy, value function, standard errors, and diagnostics
```

## Solution Modes

The `solution` option selects which Bellman object the sieve represents.

| Mode | Approximated object | Residual |
| --- | --- | --- |
| `value` | Integrated value `V(s)`. | Soft Bellman residual. |
| `q` | Choice-specific value `Q(s,a)`. | Q-Bellman residual. |
| `ev` | Expected continuation value by state and action. | Continuation consistency. |
| `policy` | Centered policy logits. | Logit optimality consistency. |
| `collocation` | Integrated value `V(s)`. | Bellman residual on deterministic collocation states. |

The strongest direct SEES-theory fit is `value`, with `q` as an equivalent
Bellman representation. The other modes are exposed for diagnostics and
numerical experiments in DDC/IRL problems.

## Basis Paths

For compact tabular problems, the implementation can build a basis over state
indices. For encoded-state problems, it can build an encoded-state basis from
the `DDCProblem` state encoder. The high-dimensional SEES simulation study
uses the encoded-state path with 81 basis functions and numerical rank 81.

## Inference

The lower-level estimator returns an `EstimationSummary` with reward
parameters, standard errors, policy, value function, likelihood, and metadata.
The metadata records the basis source, basis dimension, penalty weight,
solution type, theta-start diagnostics, Bellman violation, Bellman RMSE, and
projection diagnostics.
