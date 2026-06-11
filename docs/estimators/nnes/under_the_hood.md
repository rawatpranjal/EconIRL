# Under the Hood

NNES estimates a structural dynamic discrete choice likelihood while learning
a neural approximation to the integrated value function. The reported path is
the NPL Bellman path.

## Optimization Setup

The observed panel supplies state, action, and next-state records. The
transition law is estimated from the panel or supplied directly. Reward
features, the discount factor, the logit shock scale, the value-network
architecture, and the NPL iteration count are fixed before optimization.

NNES optimizes the finite-dimensional structural reward parameters `theta`
while using a value approximation as a nuisance object. In the reported NPL
path, the finite-state profiled continuation terms define the structural
likelihood, and the value network is trained against the same continuation
target. The network supports the approximation path; the structural estimate
remains `theta`.

## Model

The observed data are state, action, and next-state trajectories.

$$
(s_{it}, a_{it}, s_{i,t+1})
$$

The flow payoff is finite-dimensional.

$$
u_\theta(s, a) = \phi(s, a)^\top \theta
$$

NNES represents the integrated value function with a neural network.

$$
V_\psi(s) \approx V_\theta(s)
$$

Choice-specific values combine current utility and continuation value.

$$
Q_{\theta,\psi}(s,a)
= u_\theta(s,a)
  + \beta \sum_{s'} P_a(s,s') V_\psi(s')
$$

Choice probabilities follow the soft-max rule.

$$
\pi_{\theta,\psi}(a \mid s)
=
\frac{\exp(Q_{\theta,\psi}(s,a) / \sigma)}
     {\sum_b \exp(Q_{\theta,\psi}(s,b) / \sigma)}.
$$

The NPL path alternates value-network training with structural likelihood
updates. In the public results file, both synthetic cells run three outer NPL
iterations.

The value network is an approximation target, not the structural reward. The
structural object remains `theta`; the network parameters are nuisance
parameters used to represent the continuation value.

## Pseudocode

```text
Input: panel, reward features, transitions, state encoder, and value network
Initialize theta, CCPs, and value-network parameters
for each configured outer NPL iteration:
    solve or profile the continuation terms for the current CCPs
    train the value network on the profiled value target
    build Q_theta(s, a) from reward and continuation terms
    maximize the structural likelihood over theta
    update CCPs from the implied policy
return theta, policy, value approximation, standard errors, and diagnostics
```

## Operational Loop

1. Supply panel trajectories and either estimate or provide the transition
   law.
2. Keep the reward finite-dimensional, initialize structural parameters, and
   initialize the value network.
3. Use the value network to form continuation values and choice-specific
   values for each state-action pair.
4. Update choice probabilities and fit the structural likelihood along the
   NPL Bellman path.
5. Repeat the outer loop, then report reward parameters, standard errors,
   policy, value, Q, likelihood, and value-network diagnostics.

The important operational difference from NFXP is where the computation sits.
NFXP solves the exact Bellman fixed point inside each likelihood evaluation.
NNES trains a neural nuisance value object and uses that approximation inside
the policy-iteration likelihood path.

## Finite-State Profiled Path

The package simulation study uses a finite-state version of the NPL argument. For a
fixed CCP iterate `P`, the policy-evaluation value is affine in the structural
parameters:

$$
W_\theta[P] = W_z(P)\theta + W_e(P).
$$

EconIRL solves the profiled components `W_z(P)` and `W_e(P)` exactly on the
finite synthetic cells, then optimizes the structural likelihood through the
choice-specific values implied by those components. The value network is
trained on the same profiled target and reported through metadata such as
`v_loss_per_outer`; in the current finite-state study, it is also a
diagnostic for the large-state approximation path rather than the only object
driving the likelihood.

This is why the simulation study can check reward, policy, value, Q, and
counterfactual recovery directly against known oracle objects.

## Bellman Options

| Option | Meaning | Study role |
| --- | --- | --- |
| `bellman="npl"` | NPL Bellman with Hotz-Miller correction. | Primary NNES path. |
| `bellman="nfxp"` | Neural soft-Bellman approximation. | Diagnostic variant. |

The NPL path is the reported path because it is the path used by the
numerical checks and the results file. The NFXP variant is useful for
experiments but does not carry the same orthogonality interpretation.

## Anchoring

NNES uses an anchored value-network target by default, with `anchor_state=0`.
The normalization subtracts the network value at the anchor state, so the
neural value object satisfies the analogue of `V(anchor) = 0`.

This matters because logit choice probabilities depend on value differences,
not the absolute level of the value function. Without anchoring, high-discount
problems can have a nearly flat value-level direction, producing numerical
drift. Anchoring removes that redundant level while leaving CCPs, likelihoods,
and structural reward parameters unchanged.

## State Encoding

The wrapper builds a one-dimensional normalized state encoder for tabular bus
data. The lower-level simulation path can use richer encoded-state objects
from the synthetic data-generating process. The high-dimensional primary cell has 81 states, a
16-dimensional encoded state, and 32 reward parameters.

## Inference

The lower-level estimator returns an `EstimationSummary` with reward
parameters, standard errors, policy, value function, likelihood, and metadata.
The metadata records `profile_mode="exact_finite_state_npl"`, the number of
outer NPL iterations, the final CCPs, profiled choice values, the profiled
value function, and value-network loss by outer iteration.

The standard-error calculation belongs to the NPL path. The legacy neural-NFXP
Bellman-residual path can fit a model, but approximation error enters the score
directly there, so it is not the reported inference study.
