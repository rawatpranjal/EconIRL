Rust Bus Post-Estimation Diagnostics
======================================

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

.. code-block:: text

                                 NFXP           CCP
   ================================================
                theta_c      0.0010**    -0.0019***
                                (0.0004)      (0.0000)
                     RC     3.0724***     2.1062***
                                (0.0747)      (0.0000)
   ------------------------------------------------
           Observations         9,410         9,410
         Log-Likelihood     -1,900.33     -2,047.56
                    AIC       3,804.7       4,099.1
                    BIC       3,819.0       4,113.4
   ================================================
   Note: * p<0.10, ** p<0.05, *** p<0.01

The ``etable`` function produces a stargazer-style table with estimates, standard errors in parentheses, and significance stars. It also supports LaTeX and HTML output through the ``output`` argument.

Hypothesis tests
-----------------

The Vuong test compares two non-nested models by evaluating their per-observation log-likelihood differences. A positive z-statistic means the first model fits the data better.

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

.. code-block:: text

   z = 8.8291, p = 0.0000, direction = model_1

With a z-statistic of 8.83 and a p-value below 0.001, the test strongly favors NFXP over CCP on this dataset.

The ``likelihood_ratio_test`` and ``score_test`` functions test nested restrictions. The LR test takes two fitted summaries and computes the statistic directly.

.. code-block:: python

   from econirl.inference import likelihood_ratio_test

   lr = likelihood_ratio_test(restricted=ccp._result, unrestricted=nfxp._result)
   print(f"LR = {lr['statistic']:.2f}, df = {lr['df']}, p = {lr['p_value']:.4f}")

Prediction quality
-------------------

Three metrics measure how well each model predicts the observed choices.

.. code-block:: python

   from econirl.inference import brier_score, kl_divergence, efron_pseudo_r_squared

   bs_nfxp = brier_score(jnp.array(nfxp.policy_), obs_states, obs_actions)
   bs_ccp  = brier_score(jnp.array(ccp.policy_), obs_states, obs_actions)
   print(f"Brier score NFXP: {bs_nfxp['brier_score']:.6f}")
   print(f"Brier score CCP:  {bs_ccp['brier_score']:.6f}")

.. code-block:: text

   Brier score NFXP: 0.097296
   Brier score CCP:  0.101451

.. code-block:: python

   kl_nfxp = kl_divergence(data_ccps, jnp.array(nfxp.policy_), state_freq)
   kl_ccp  = kl_divergence(data_ccps, jnp.array(ccp.policy_), state_freq)
   print(f"KL divergence NFXP: {kl_nfxp['kl_divergence']:.6f}")
   print(f"KL divergence CCP:  {kl_ccp['kl_divergence']:.6f}")

.. code-block:: text

   KL divergence NFXP: 0.004101
   KL divergence CCP:  0.019748

.. code-block:: python

   er_nfxp = efron_pseudo_r_squared(jnp.array(nfxp.policy_), obs_states, obs_actions)
   er_ccp  = efron_pseudo_r_squared(jnp.array(ccp.policy_), obs_states, obs_actions)
   print(f"Efron R² NFXP: {er_nfxp['efron_r_squared']:.4f}")
   print(f"Efron R² CCP:  {er_ccp['efron_r_squared']:.4f}")

.. code-block:: text

   Efron R² NFXP:  0.0009
   Efron R² CCP:  -0.0417

NFXP achieves a lower Brier score and KL divergence on every metric, consistent with the Vuong test result. The Efron R-squared values are small because the replacement event is rare, so even good models produce modest values in absolute terms.

.. list-table:: Prediction Metrics Summary
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

Specification tests
--------------------

The CCP consistency test compares the empirical choice probabilities to the model-implied probabilities using a Pearson chi-squared statistic.

.. code-block:: python

   from econirl.inference import ccp_consistency_test

   test = ccp_consistency_test(
       data_ccps, jnp.array(nfxp.policy_),
       jnp.array(state_counts),
       num_estimated_params=nfxp._result.num_parameters,
   )
   print(f"chi2 = {test['statistic']:.2f}, df = {test['df']}, p = {test['p_value']:.4f}")

.. code-block:: text

   chi2 = 68.81, df = 50, p = 0.0399

The NFXP model produces a chi-squared statistic of 68.81 with 50 degrees of freedom and a p-value of 0.04. The marginal rejection suggests a reasonable but imperfect fit, which is typical for structural models estimated on real data.

Reward comparison
------------------

The EPIC distance measures how different two reward functions are after removing potential-based shaping and scale differences. A distance of zero means the two rewards induce the same optimal policy for any transition dynamics.

.. code-block:: python

   from econirl.inference import epic_distance

   # Build reward vectors from estimated parameters
   reward_nfxp = np.zeros((num_states, 2))
   reward_nfxp[:, 0] = -nfxp.params_["theta_c"] * np.arange(num_states)
   reward_nfxp[:, 1] = -nfxp.params_["RC"]

   reward_ccp = np.zeros((num_states, 2))
   reward_ccp[:, 0] = -ccp.params_["theta_c"] * np.arange(num_states)
   reward_ccp[:, 1] = -ccp.params_["RC"]

   epic = epic_distance(jnp.array(reward_nfxp), jnp.array(reward_ccp), 0.9999)
   print(f"EPIC distance:       {epic['epic_distance']:.6f}")
   print(f"Pearson correlation: {epic['pearson_correlation']:.6f}")

.. code-block:: text

   EPIC distance:       0.021809
   Pearson correlation: 0.999049

The EPIC distance between the NFXP and CCP reward vectors is 0.022 with a Pearson correlation of 0.999. The two estimators recover nearly identical reward structures despite different computational approaches.

The shaping detection function tests whether two rewards differ only by a potential function and recovers the potential if they do. A synthetic example demonstrates exact recovery.

.. code-block:: python

   from econirl.inference import detect_reward_shaping

   r_base = jnp.zeros((5, 2, 5))
   phi_true = jnp.array([0.0, 1.5, -0.8, 2.3, -1.1])
   shaped = r_base + 0.99 * phi_true[None, None, :] - phi_true[:, None, None]

   result = detect_reward_shaping(r_base, shaped, 0.99)
   print(f"Is shaping:         {result['is_shaping']}")
   print(f"Relative residual:  {result['relative_residual']:.6f}")
   print(f"Recovered potential: {[f'{x:.2f}' for x in result['potential']]}")
   print(f"True potential:      {[f'{float(x):.2f}' for x in phi_true]}")

.. code-block:: text

   Is shaping:         True
   Relative residual:  0.000000
   Recovered potential: ['0.00', '1.50', '-0.80', '2.30', '-1.10']
   True potential:      ['0.00', '1.50', '-0.80', '2.30', '-1.10']

Running the example
--------------------

.. code-block:: bash

   python examples/post_estimation_diagnostics.py

Reference
---------

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Vuong, Q. H. (1989). Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses. Econometrica, 57(2), 307-333.

Gleave, A., Dennis, M., Legg, S., Russell, S., and Leike, J. (2020). Quantifying Differences in Reward Functions. ICLR 2021.
