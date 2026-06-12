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
| GLADIUS (model-free) | 1.291 | 25.71 | n/a |

Neural MPEC drives the Bellman residual to near zero and recovers reward and value far
better than model-free GLADIUS, but sits about 8x above tabular MPEC on reward RMSE.
Sweeping `rho` over {0.1, 1, 10}, net width/depth, and collocation set (all states vs
observed-frequency-weighted) barely moves the number. The model's final NLL reaches the
true-policy NLL floor, so the likelihood is fully optimized. The gap is not optimization.

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
| 256k | 0.033 | 0.119 | 0.007 | 0.112 |

By 256k obs the neural value RMSE matches tabular (0.119 vs 0.112) and the reward RMSE
is still falling. (The 256k row is from the convergence probe, not the default sweep.)

## Takeaway

Neural MPEC is real and well-behaved. The known-`P` exact residual makes it stable
(soft Bellman is a contraction for fixed reward, no double-sampling), it is consistent,
and it dominates the model-free neural cousin. Its cost is statistical efficiency under
correct linear specification, which is exactly the price of not knowing the reward is
linear. The place it should win is a DGP where the true reward is **not** linear in the
features, where pooling into 4 parameters is misspecification and the flexible reward
pays off. That is the natural next experiment.
