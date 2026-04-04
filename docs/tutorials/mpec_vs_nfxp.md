# MPEC vs NFXP Parameter Recovery

| | |
|---|---|
| **Estimators** | NFXP-NK (SA-then-NK hybrid), MPEC (SLSQP) |
| **Environment** | Rust bus engine, varying bins and discount factors |
| **Key finding** | Both recover identical MLE. MPEC-SLSQP is 2 to 22 times faster depending on the scenario. |

## Background

NFXP (Rust 1987) nests a full Bellman solve inside each optimizer step. The hybrid polyalgorithm (Iskhakov, Rust and Schjerning 2016) starts with successive approximation and switches to Newton-Kantorovich iterations for quadratic convergence near the fixed point. This combination makes NFXP-NK dramatically faster than the pure contraction NFXP-SA that Su and Judd (2012) used in their original comparison. MPEC (Su and Judd 2012) treats the value function as decision variables alongside the structural parameters and enforces the Bellman equation as an equality constraint. Our implementation uses scipy SLSQP with an analytical constraint Jacobian computed via JAX.

Iskhakov, Rust and Schjerning (2016) showed that Su and Judd's reported MPEC speed advantage was an artifact of comparing against NFXP-SA rather than NFXP-NK. With NFXP-NK and KNITRO, the two methods were comparable on small problems and NFXP-NK dominated on larger problems. MPEC with KNITRO also failed to converge at high discount factors. Our SLSQP implementation does not exhibit these convergence failures, but the speed comparison depends on the state space size and discount factor.

## Setup

The comparison runs a grid over three discount factors (0.95, 0.99, 0.9999) and two state space sizes (90 and 200 mileage bins). Each cell simulates 200 buses over 100 periods with true parameters theta_c equal to 0.001 and RC equal to 3.0. NFXP uses the hybrid inner solver with tolerance 1e-12 and switch tolerance 1e-3. MPEC uses scipy SLSQP with constraint tolerance 1e-8.

## Code

The full script is at ``examples/rust-bus-engine/mpec_vs_nfxp_mc.py``.

```python
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mpec import MPECEstimator, MPECConfig

nfxp = NFXPEstimator(
    inner_solver="hybrid",
    inner_tol=1e-12,
    switch_tol=1e-3,
)

mpec = MPECEstimator(
    config=MPECConfig(solver="slsqp", max_iter=500),
)
```

## Results

One replication per cell. Log-likelihoods are identical to machine precision in all cells, confirming that both methods solve the same optimization problem.

| beta | N | NFXP time (s) | MPEC time (s) | Speedup |
|---|---|---|---|---|
| 0.95 | 90 | 22.8 | 3.7 | 6.2x |
| 0.95 | 200 | 32.8 | 3.6 | 9.1x |
| 0.99 | 90 | 22.8 | 13.8 | 1.6x |
| 0.99 | 200 | 38.8 | 4.0 | 9.6x |
| 0.9999 | 90 | 67.9 | 18.0 | 3.8x |
| 0.9999 | 200 | 147.1 | 6.8 | 21.7x |

Both methods converge in every cell with identical log-likelihoods.

## Discussion

MPEC-SLSQP is consistently faster than NFXP-NK across all scenarios tested here, with the speed advantage growing as the state space increases. At 200 bins and beta equal to 0.9999, MPEC is 22 times faster because the NFXP inner loop must run many contraction iterations on a 200-dimensional vector before the Newton-Kantorovich phase can begin. MPEC avoids this inner loop entirely by treating V as free variables and letting the SLSQP optimizer enforce the Bellman constraint directly.

These results differ from Iskhakov, Rust and Schjerning (2016), who found NFXP-NK comparable to or faster than MPEC-KNITRO. The difference is likely due to the solver: scipy SLSQP handles the Bellman equality constraint differently from KNITRO's interior-point method. In particular, Iskhakov et al. reported frequent convergence failures of MPEC-KNITRO at high discount factors (only 27 of 1250 runs converged at beta equal to 0.9995 with AMPL/KNITRO). Our SLSQP implementation converges reliably across all scenarios. However, Iskhakov et al. also showed that NFXP-NK dominates MPEC when the state space exceeds roughly 500 states, where MPEC's optimization over N+K variables becomes exponentially expensive. For problems with thousands of states, NFXP-NK or sieve-based methods like SEES are the better choice.

The bottom line is that both methods produce identical estimates and neither is universally faster. MPEC-SLSQP is a strong default for problems with moderate state spaces and any discount factor. NFXP-NK is the safer choice for very large state spaces or when robustness is paramount.
