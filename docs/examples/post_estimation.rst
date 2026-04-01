Rust Bus Post-Estimation Diagnostics
======================================

.. image:: /_static/rust_bus_counterfactual.png
   :alt: Counterfactual analysis of the Rust bus engine replacement problem showing replacement probability and value function under varying costs.
   :width: 100%

This example demonstrates the full post-estimation diagnostic suite on the Rust (1987) bus engine dataset. After estimating two models on the same data, the diagnostics compare fit, test hypotheses, and measure reward equivalence. The 10 tools shown here are all available from ``econirl.inference``.

Quick start
-----------

Fit NFXP and CCP on the bus engine data, then compare them side by side with ``etable``.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rust_bus
   from econirl.inference import etable

   df = load_rust_bus()

   nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp  = CCP(discount=0.9999, num_policy_iterations=3).fit(df, state="mileage_bin", action="replaced", id="bus_id")

   print(etable(nfxp._result, ccp._result, model_names=["NFXP", "CCP"]))

The ``etable`` function produces a stargazer-style table with estimates, standard errors in parentheses, and significance stars. It also supports LaTeX and HTML output through the ``output`` argument.

.. list-table:: Model Comparison
   :header-rows: 1

   * -
     - NFXP
     - CCP
   * - theta_c
     - 0.0010**
     - -0.0019***
   * -
     - (0.0004)
     - (0.0000)
   * - RC
     - 3.0724***
     - 2.1062***
   * -
     - (0.0747)
     - (0.0000)
   * - Log-Likelihood
     - -1,900.33
     - -2,047.56
   * - AIC
     - 3,804.7
     - 4,099.1

Hypothesis tests
-----------------

The Vuong test compares two non-nested models by evaluating their per-observation log-likelihood differences. A positive z-statistic means the first model fits the data better. With a z-statistic of 8.83 and a p-value below 0.001, the test strongly favors NFXP over CCP on this dataset.

.. code-block:: python

   from econirl.inference import vuong_test
   import jax.numpy as jnp

   obs_states = jnp.array(df["mileage_bin"].values, dtype=jnp.int32)
   obs_actions = jnp.array(df["replaced"].values, dtype=jnp.int32)

   vt = vuong_test(
       jnp.array(nfxp.policy_), jnp.array(ccp.policy_),
       obs_states, obs_actions,
   )
   print(f"z = {vt['statistic']:.4f}, p = {vt['p_value']:.4f}, direction = {vt['direction']}")

.. list-table:: Hypothesis Test Results
   :header-rows: 1

   * - Test
     - Statistic
     - p-value
     - Interpretation
   * - Vuong (NFXP vs CCP)
     - z = 8.83
     - < 0.001
     - NFXP fits better

The ``likelihood_ratio_test`` and ``score_test`` functions test nested restrictions. The LR test requires two fitted models where the restricted model is a special case of the unrestricted. The Score test requires only the restricted model plus the gradient of the unrestricted log-likelihood at the restricted parameters.

Prediction quality
-------------------

Three metrics measure how well each model predicts the observed choices. The Brier score is the average squared difference between predicted probabilities and the actual choices, where lower is better. The KL divergence measures how much information is lost when the model distribution is used to approximate the data distribution. The Efron pseudo R-squared is a variance-ratio analog of the linear regression R-squared.

.. code-block:: python

   from econirl.inference import brier_score, kl_divergence, efron_pseudo_r_squared

   bs = brier_score(jnp.array(nfxp.policy_), obs_states, obs_actions)
   kl = kl_divergence(data_ccps, jnp.array(nfxp.policy_), state_freq)
   er = efron_pseudo_r_squared(jnp.array(nfxp.policy_), obs_states, obs_actions)

.. list-table:: Prediction Metrics
   :header-rows: 1

   * - Metric
     - NFXP
     - CCP
   * - Brier Score
     - 0.0973
     - 0.1015
   * - KL Divergence
     - 0.0041
     - 0.0197
   * - Efron R-squared
     - 0.0009
     - -0.0417

NFXP achieves a lower Brier score and KL divergence, consistent with the Vuong test result. The Efron R-squared values are small because the replacement event is rare, so even good models produce modest R-squared values in absolute terms.

Specification tests
--------------------

The CCP consistency test compares the empirical choice probabilities to the model-implied probabilities using a Pearson chi-squared statistic. A small p-value indicates that the model does not perfectly reproduce the observed choice frequencies conditional on state.

.. code-block:: python

   from econirl.inference import ccp_consistency_test

   test = ccp_consistency_test(
       data_ccps, jnp.array(nfxp.policy_),
       jnp.array(state_counts),
       num_estimated_params=nfxp._result.num_parameters,
   )
   print(f"chi2 = {test['statistic']:.2f}, df = {test['df']}, p = {test['p_value']:.4f}")

The NFXP model produces a chi-squared statistic of 68.81 with 50 degrees of freedom and a p-value of 0.04. The marginal rejection suggests a reasonable but imperfect fit, which is typical for structural models estimated on real data.

Reward comparison
------------------

The EPIC distance measures how different two reward functions are after removing potential-based shaping and scale differences. A distance of zero means the two rewards induce the same optimal policy for any transition dynamics. The shaping detection function tests whether two rewards differ only by a potential function and recovers the potential if they do.

.. code-block:: python

   from econirl.inference import epic_distance, detect_reward_shaping

   epic = epic_distance(jnp.array(reward_nfxp), jnp.array(reward_ccp), 0.9999)
   print(f"EPIC distance = {epic['epic_distance']:.6f}")
   print(f"Correlation   = {epic['pearson_correlation']:.6f}")

The EPIC distance between the NFXP and CCP reward vectors is 0.022 with a Pearson correlation of 0.999. The two estimators recover nearly identical reward structures despite different computational approaches.

For the shaping detection tool, a synthetic example demonstrates exact recovery of a known potential function.

.. code-block:: python

   r_base = jnp.zeros((5, 2, 5))
   phi_true = jnp.array([0.0, 1.5, -0.8, 2.3, -1.1])
   shaped = r_base + 0.99 * phi_true[None, None, :] - phi_true[:, None, None]

   result = detect_reward_shaping(r_base, shaped, 0.99)
   print(f"Is shaping: {result['is_shaping']}")  # True
   print(f"Recovered: {result['potential']}")     # matches phi_true

Running the example
--------------------

.. code-block:: bash

   python examples/post_estimation_diagnostics.py

Reference
---------

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Vuong, Q. H. (1989). Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses. Econometrica, 57(2), 307-333.

Gleave, A., Dennis, M., Legg, S., Russell, S., and Leike, J. (2020). Quantifying Differences in Reward Functions. ICLR 2021.
