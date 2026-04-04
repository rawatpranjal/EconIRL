# NNES vs NFXP Parameter Recovery

| | |
|---|---|
| **Estimators** | NFXP (exact Bellman), NNES-NPL (neural V approximation) |
| **Environment** | Rust bus engine, 90 bins, beta 0.99 |
| **Key finding** | NNES matches NFXP on log-likelihood and RC recovery. The small operating cost is harder for the neural approximation. |

## Background

NFXP solves the Bellman equation exactly at each optimizer step, which gives it the best possible precision for tabular state spaces. NNES (Nguyen 2025) replaces the exact Bellman solve with a neural network that approximates the value function V(s). The key theoretical insight is that the NPL variant of NNES has a zero-Jacobian property: first-order errors in the V approximation drop out of the structural parameter score. This means NNES achieves the same asymptotic efficiency as NFXP despite using an approximate value function. The comparison here uses the standard Rust bus engine where NFXP serves as the oracle benchmark.

## Setup

The environment uses 90 mileage bins and a discount factor of 0.99. Each replication simulates 200 buses over 100 periods. The NFXP estimator uses the hybrid inner solver with exact Bellman convergence. The NNES estimator uses a two-layer MLP with 32 hidden units, trained for 500 epochs per outer iteration with 3 outer NPL iterations.

## Code

The full Monte Carlo script is at ``examples/rust-bus-engine/nnes_vs_nfxp_mc.py``.

```python
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.nnes import NNESEstimator

nfxp = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-12)

nnes = NNESEstimator(
    hidden_dim=32,
    v_epochs=500,
    n_outer_iterations=3,
)
```

## Results

Five Monte Carlo replications with true parameters theta_c equal to 0.001 and RC equal to 3.0.

| Metric | NFXP | NNES |
|---|---|---|
| Bias (theta_c) | 0.0001 | 0.0061 |
| Bias (RC) | 0.0003 | 0.0007 |
| RMSE (theta_c) | 0.0004 | 0.6270 |
| RMSE (RC) | 0.0629 | 0.0409 |
| Mean log-likelihood | -4203.34 | -4203.29 |
| Mean wall time (seconds) | 22.3 | 51.0 |

## Discussion

The log-likelihoods are nearly identical, confirming that both estimators find the same quality of fit. The replacement cost RC is recovered well by both methods, with NNES actually achieving lower RMSE (0.041 vs 0.063). The operating cost theta_c, however, shows much higher variance under NNES. This parameter is three orders of magnitude smaller than RC (0.001 vs 3.0), making it harder for the neural network to resolve the fine-grained mileage gradient in the value function.

The difficulty is inherent to the tabular setting where NFXP already works perfectly. NFXP solves the Bellman equation to machine precision, so even tiny parameters are identified exactly through the likelihood curvature. The neural V approximation introduces noise that is small relative to RC but large relative to theta_c. The NPL zero-Jacobian property ensures that this noise does not bias the standard errors, but it does not eliminate the noise itself.

The value of NNES emerges in continuous or high-dimensional state spaces where NFXP cannot operate because it requires enumerating all states. In those settings, the neural approximation is the only feasible path to structural MLE with valid inference. On tabular problems like this 90-state bus engine, NFXP remains the gold standard.
