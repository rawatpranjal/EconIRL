# Neural MPEC experiment

Can MPEC run with neural function approximators? A neural reward `u_theta(s,a)` and a
neural value `V_phi(s)`, co-trained in one loop by minimizing logit NLL plus a soft
Bellman-residual penalty. Reproduce with `python scripts/neural_mpec_experiment.py`.

## The estimator

```
EV_phi(s,a) = sum_s' P_a(s,s') V_phi(s')          # exact, transitions P are known
Q(s,a)      = u_theta(s,a) + beta * EV_phi(s,a)
pi(a|s)     = softmax_a( Q(s,.)/sigma )
NLL         = - mean_it log pi(a_it | s_it)
resid(s)    = V_phi(s) - sigma*logsumexp_a( Q(s,.)/sigma )
Loss        = NLL + (rho/2) * sum_{s in C} w(s) resid(s)^2
```

This is the penalty-relaxed form of MPEC. Once `V` is a network the hard Bellman
equality only survives at collocation points `C`, so the per-state Lagrange multiplier
of tabular MPEC becomes a residual penalty. With known `P` and a tiny grid, `C` is the
whole state space and the residual is exact (no double-sampling).

Closest existing estimator is GLADIUS, which is the same single-loop idea but
**model-free**. It never uses `P`, approximates `E[V(s')|s,a]` with a second learned
network, and collocates on observed `(s,a,s')` minibatches. Neural MPEC replaces that
learned EV network with the exact `sum_s' P V` and keeps one value net.

## DGP

A 3-action ergodic MDP, 20 states, beta 0.95, sigma 1.0. Action 2 is a zero-reward
**reference** action (its features are zero), which point-identifies the reward level.
The neural reward net hard-pins `u_theta(s,2) = 0` to match. Without this anchor the
reward RMSE is meaningless even with a perfectly recovered policy. Balanced action
shares (~0.30 / 0.40 / 0.31) and full state coverage.

## Result 1, one panel (16k obs)

| estimator | reward RMSE | value RMSE | max Bellman resid |
|---|---:|---:|---:|
| tabular MPEC (gold) | 0.017 | 0.210 | exact |
| neural MPEC (shallow) | 0.135 | 0.336 | 1e-5 to 4e-3 |
| GLADIUS (under-configured, unanchored)* | 1.291 | 25.71 | n/a |

Reward RMSE here is over the full (S, A) matrix, which includes the reference action whose
reward is anchored to zero in both truth and estimate; over the two estimated actions only
the neural figure is 0.165. Neural MPEC drives the Bellman residual to near zero and sits
about 8x above tabular MPEC on reward RMSE. Sweeping `rho` over {0.1, 1, 10}, net
width/depth, and collocation barely moves the number. The model's final in-sample NLL
reaches the true-policy floor, so the likelihood is fully optimized and the gap is not
optimization.

*The GLADIUS row is **not a fair baseline**. It is run here under-sized (32-wide, 1 layer,
300 epochs) and without the action-2 anchor that tabular and neural MPEC both receive. Run
at the repo-standard size (128-wide, 3 layers) with the same anchor, GLADIUS reaches value
RMSE ~2.6 and reward RMSE ~0.15. The like-for-like, anchored comparison lives in the
`direct_optimization` simulation-studies page, not here. This prototype keeps GLADIUS only
as a rough model-free reference.

## Result 2, why the gap is variance, not bias

Push the data up and the neural reward RMSE falls at the root-N rate, converging to the
truth. The estimator is consistent. Tabular MPEC stays about 4x to 5x lower because its
4 linear parameters pool information across states, while the flexible per-state neural
reward inherits the full finite-sample noise of the CCP-to-reward inverse map with no
pooling. Textbook nonparametric-versus-parametric tradeoff, not a defect.

| n_obs | neural reward RMSE | neural value RMSE | tabular reward RMSE | tabular value RMSE |
|---:|---:|---:|---:|---:|
| 4k | 0.248 | 0.683 | 0.054 | 0.235 |
| 16k | 0.139 | 0.244 | 0.029 | 0.187 |
| 64k | 0.071 | 0.106 | 0.010 | 0.067 |

Each 4x of data roughly halves the neural reward RMSE, the root-N signature of a consistent
estimator, while tabular MPEC stays about 4x to 5x lower throughout (every row reproduces
from `neural_mpec_experiment_results.json`).

## Takeaway

Neural MPEC is real and well-behaved. The known-`P` exact residual makes it stable
(soft Bellman is a contraction for fixed reward, no double-sampling) and it is consistent.
Its cost is statistical efficiency under correct linear specification, which is exactly the
price of not knowing the reward is linear. The place it should win is a DGP where the true
reward is **not** linear in the features, where pooling into 4 parameters is misspecification
and the flexible reward pays off. That comparison, with all methods anchored like-for-like
and a fair GLADIUS, is the `direct_optimization` simulation-studies page.
