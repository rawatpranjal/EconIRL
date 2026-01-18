Inference
=========

.. module:: econirl.inference

Statistical inference for estimated models: standard errors, confidence intervals,
hypothesis tests, and goodness-of-fit measures.

EstimationResult
----------------

.. autoclass:: econirl.inference.results.EstimationResult
   :members:
   :undoc-members:
   :show-inheritance:

   Container for estimation results with rich inference capabilities.

   **Example:**

   .. code-block:: python

      # After estimation
      result = estimator.estimate(panel, utility, problem, transitions)

      # Summary table (StatsModels-style)
      print(result.summary())

      # Access components
      print(f"Parameters: {result.parameters}")
      print(f"Standard errors: {result.standard_errors}")
      print(f"Log-likelihood: {result.log_likelihood}")

      # Confidence intervals
      ci_low, ci_high = result.confidence_interval(alpha=0.05)

      # Hypothesis tests
      t_stats = result.t_statistics
      p_values = result.p_values

      # Export
      df = result.to_dataframe()
      latex = result.to_latex(caption="My Results")

GoodnessOfFit
-------------

.. autoclass:: econirl.inference.results.GoodnessOfFit
   :members:
   :undoc-members:

   Goodness-of-fit measures for model evaluation.

   **Attributes:**

   - ``log_likelihood``: Log-likelihood at estimated parameters
   - ``aic``: Akaike Information Criterion
   - ``bic``: Bayesian Information Criterion
   - ``pseudo_r_squared``: McFadden's pseudo R²
   - ``prediction_accuracy``: Fraction of correctly predicted choices

Standard Errors
---------------

.. automodule:: econirl.inference.standard_errors
   :members:
   :undoc-members:
   :show-inheritance:

Identification
--------------

.. automodule:: econirl.inference.identification
   :members:
   :undoc-members:
   :show-inheritance:

   Tools for diagnosing identification issues.

   **Example:**

   .. code-block:: python

      # Check identification diagnostics in result
      print(f"Hessian condition number: {result.identification.condition_number}")
      print(f"Min eigenvalue: {result.identification.min_eigenvalue}")

      if result.identification.is_identified:
          print("Model appears well-identified")
      else:
          print("Warning: Potential identification issues")
