Estimation
==========

.. module:: econirl.estimation

Estimators for dynamic discrete choice models. Each estimator takes panel data,
a utility specification, and structural parameters, and returns an
``EstimationResult`` with parameter estimates and inference.

NFXPEstimator
-------------

.. autoclass:: econirl.estimation.nfxp.NFXPEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   Nested Fixed Point estimator (Rust, 1987).

   **Algorithm:**

   1. Outer loop: Search over parameters θ
   2. Inner loop: Solve Bellman equation to get value function V(s; θ)
   3. Compute choice probabilities P(a|s; θ) from V
   4. Maximize log-likelihood of observed choices

   **Example:**

   .. code-block:: python

      from econirl import NFXPEstimator, LinearUtility, RustBusEnvironment
      from econirl.simulation import simulate_panel

      env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)
      panel = simulate_panel(env, n_individuals=500, n_periods=100)
      utility = LinearUtility.from_environment(env)

      estimator = NFXPEstimator(
          se_method="asymptotic",
          optimizer="L-BFGS-B",
          verbose=True,
      )

      result = estimator.estimate(
          panel=panel,
          utility=utility,
          problem=env.problem_spec,
          transitions=env.transition_matrices,
      )

      print(result.summary())

CCPEstimator
------------

.. autoclass:: econirl.estimation.ccp.CCPEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   Conditional Choice Probability estimator (Hotz & Miller, 1993).

   **Algorithm:**

   1. First stage: Estimate choice probabilities P(a|s) from data
   2. Second stage: Invert to get value differences, estimate parameters

   Faster than NFXP but requires larger samples for first-stage precision.

Base Classes
------------

.. automodule:: econirl.estimation.base
   :members:
   :undoc-members:
   :show-inheritance:

Estimation Options
------------------

**Standard Error Methods:**

- ``"asymptotic"``: Inverse Hessian (default)
- ``"bootstrap"``: Bootstrap standard errors
- ``"robust"``: Heteroskedasticity-robust (sandwich)

**Optimizers:**

- ``"L-BFGS-B"``: Limited-memory BFGS with bounds (default)
- ``"BFGS"``: BFGS without bounds
- ``"Newton-CG"``: Newton conjugate gradient

**Convergence:**

- ``inner_tol``: Tolerance for value function iteration (default: 1e-10)
- ``outer_tol``: Tolerance for parameter optimization (default: 1e-6)
- ``max_iter``: Maximum outer iterations (default: 1000)
