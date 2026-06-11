# MPEC Paper Context

Primary sources: Su and Judd (2012) for mathematical programming with
equilibrium constraints, and Iskhakov et al. (2016) for the computational
critique and comparison with NFXP.

Public citations live in `../../../docs/references.md`. Broader paper routing
lives in `../../papers/index.md`.

Internal context to preserve here: constrained likelihood formulation,
Bellman equality residuals, value variables, solver tolerance choices,
high-discount behavior, Hessian and covariance handling, and NFXP equivalence.

## Paper-To-Package Translation

Su and Judd (2012) supplies the constrained formulation: optimize over
structural parameters and equilibrium objects jointly, while imposing the
Bellman fixed point as an equality constraint. In package terms, the joint
optimization variable is `(theta, V)`.

The package keeps the paper's key equivalence: at any feasible point, MPEC and
NFXP evaluate the same dynamic discrete choice likelihood. The difference is
numerical geometry, not the structural target.

The current validation differs from an empirical Su-Judd application:

- the canonical cell is synthetic and supplies true transitions;
- the value, Q, reward, policy, and counterfactual oracle objects are known;
- the SLSQP path gates explicitly on final Bellman constraint violation;
- robust standard errors are checked for finiteness as part of the claim.

Iskhakov et al. (2016) is the cautionary source. MPEC is clean and expressive,
but the constrained problem can become fragile as the state dimension grows or
the discount factor approaches one. The package therefore treats MPEC as a
moderate-tabular structural estimator and keeps a small high-beta smoke guard
under `validation/estimators/mpec/high_beta_smoke.py`.
