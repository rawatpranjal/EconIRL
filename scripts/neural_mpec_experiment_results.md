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

Reward RMSE is over the estimated actions {0, 1} only; the reference action is anchored to
zero in both truth and estimate, so it is excluded (including it is a free win). Value RMSE
is against the soft-Bellman oracle value. GLADIUS is run at the repo-standard size (128 wide,
3 layers, 500 epochs) with the same action-2 anchor the MPEC methods receive, so it is a fair
baseline.

| estimator | reward RMSE | value RMSE | max Bellman resid |
|---|---:|---:|---:|
| tabular MPEC (gold) | 0.021 | 0.210 | exact |
| neural MPEC (shallow) | 0.165 | 0.335 | 1e-5 to 4e-3 |
| GLADIUS (model-free) | 0.185 | 2.608 | n/a |

Neural MPEC drives the Bellman residual to near zero and sits about 8x above tabular MPEC on
reward RMSE; the `rho` sweep over {0.1, 1, 10}, net width/depth, and collocation barely move
the number. Its final in-sample NLL reaches the true-policy floor, so the likelihood is fully
optimized and the remaining gap is finite-sample, not optimization.

GLADIUS recovers a comparable reward (0.185) but a far worse value function (2.6 against 0.21
and 0.34). That value gap is the honest cost of being model-free: GLADIUS never uses the
known transitions and must learn the expected-continuation operator from data, whereas the
two MPEC methods compute it exactly. Its reward also sits in a different, model-free gauge, so
that number is a model-free reference, not a like-for-like structural figure.

## Result 2, why the gap is variance, not bias

Push the data up and the neural reward RMSE falls at the root-N rate, converging to the
truth: the estimator is consistent. Tabular MPEC stays about 4x to 5x lower because its
4 linear parameters pool information across states, while the flexible per-state neural
reward inherits the finite-sample noise of the choice-probability-to-reward inverse map with
no pooling. Textbook nonparametric-versus-parametric tradeoff, not a defect. (Reward RMSE
over estimated actions {0, 1}.)

| n_obs | neural reward RMSE | neural value RMSE | tabular reward RMSE | tabular value RMSE |
|---:|---:|---:|---:|---:|
| 4k | 0.304 | 0.683 | 0.066 | 0.234 |
| 16k | 0.170 | 0.244 | 0.035 | 0.186 |
| 64k | 0.087 | 0.106 | 0.012 | 0.067 |

Each 4x of data roughly halves the neural reward RMSE, the root-N signature of a consistent
estimator, while tabular MPEC stays about 4x to 5x lower throughout. Every row reproduces
from `neural_mpec_experiment_results.json`. (The scaling sweep draws a fresh panel, so the
neural reward at 16k, 0.170, is a separate draw from the 0.165 in Result 1.)

## Takeaway

Neural MPEC is real and well-behaved. The known-`P` exact residual makes it stable
(soft Bellman is a contraction for fixed reward, no double-sampling) and it is consistent.
Its cost is statistical efficiency under correct linear specification, which is exactly the
price of not knowing the reward is linear. The place it should win is a DGP where the true
reward is **not** linear in the features, where pooling into 4 parameters is misspecification
and the flexible reward pays off. That comparison, with all methods anchored like-for-like
and a fair GLADIUS, is the `direct_optimization` simulation-studies page.
