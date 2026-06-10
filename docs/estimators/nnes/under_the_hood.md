# Under the Hood

NNES estimates a structural dynamic discrete choice likelihood while learning
a neural approximation to the integrated value function. The certified path is
the NPL Bellman path.

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
updates. In the public artifact, both validation cells run three outer NPL
iterations.

## Bellman Options

| Option | Meaning | Validation role |
| --- | --- | --- |
| `bellman="npl"` | NPL Bellman with Hotz-Miller correction. | Certified NNES path. |
| `bellman="nfxp"` | Neural soft-Bellman approximation. | Diagnostic variant. |

The NPL path is the release surface because it is the path used by the
known-truth gates and the artifact. The NFXP variant is useful for experiments
but does not carry the same orthogonality interpretation.

## State Encoding

The wrapper builds a one-dimensional normalized state encoder for tabular bus
data. The lower-level validation path can use richer encoded-state objects
from the known-truth DGP. The high-dimensional primary cell has 81 states, a
16-dimensional encoded state, and 32 reward parameters.

## Inference

The lower-level estimator returns an `EstimationSummary` with reward
parameters, standard errors, policy, value function, likelihood, and metadata.
The metadata records the profile mode, number of outer NPL iterations, and
value-network loss by outer iteration.
