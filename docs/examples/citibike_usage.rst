Citibike Daily Ride Frequency
==============================

This example applies structural estimation to a daily ride frequency problem for Citibike members. Each day a member decides whether to take a bikeshare trip. The state captures the day type (weekday or weekend) and recent usage intensity measured as the number of rides in the last seven days. The model recovers preferences over weekend riding, habit strength from recent usage, and the per-ride cost from observed behavior.

The environment has 8 states (2 day types by 4 recent usage buckets: zero rides, one to two rides, three to five rides, and six or more rides in the past week). Riding increases the usage bucket by one and not riding decreases it by one, capturing habit formation and decay. Day type transitions reflect the 5/7 weekday probability in a standard week. Three features enter the ride utility: a weekend indicator, normalized recent usage intensity, and a ride cost indicator.

.. image:: /_static/citibike_usage_overview.png
   :alt: Daily Citibike usage patterns showing weekday commuting peaks, weekend leisure riding, and habit formation in ride frequency over time.
   :width: 100%

If real Citibike data has not been downloaded, the loader generates synthetic panels from the environment with default parameters. To use real data, run the download script first.

Quick start
-----------

.. code-block:: python

   from econirl.environments.citibike_usage import CitibikeUsageEnvironment
   from econirl.datasets.citibike_usage import load_citibike_usage

   env = CitibikeUsageEnvironment(discount_factor=0.95)
   panel = load_citibike_usage(as_panel=True)

To download real data first:

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01

Estimation
----------

Three estimators recover the utility parameters from 800 training members over 90 days (72,000 observations). The true parameters are a weekend effect of negative 0.30, a habit strength of 0.80, and a ride cost of negative 0.50.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nfxp_result = NFXPEstimator(se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   ccp_result = CCPEstimator(num_policy_iterations=20, se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.1, outer_max_iter=300)).estimate(panel, utility, env.problem_spec, transitions)

.. list-table:: Parameter Recovery (800 riders, 90 days)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP (K=20)
     - MCE-IRL
   * - weekend_effect
     - -0.3000
     - -0.3072
     - -0.2957
     - -0.3072
   * - habit_strength
     - 0.8000
     - 0.8111
     - 0.6679
     - 0.8111
   * - ride_cost
     - -0.5000
     - -0.5073
     - -0.3159
     - -0.5073

NFXP and MCE-IRL recover all three parameters within 2 percent of their true values and produce identical estimates. The Hessian condition number for NFXP is 74.9, indicating clean identification. CCP underestimates habit strength by 17 percent and ride cost by 37 percent. The NPL fixed point converges to a different basin on this problem, consistent with known instability of the NPL algorithm when the discount factor is high and habit dynamics create strong serial dependence.

The weekend effect is negative 0.31, meaning riders are less likely to ride on weekends. This is consistent with a commuter-dominated member base. The habit strength of 0.81 means that past riding strongly reinforces future riding, creating a positive feedback loop. The ride cost of negative 0.51 captures the disutility of each trip including time, effort, and any usage fees.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NFXP (Nested Fixed Point)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
         weekend_effect    -0.3072***    -0.2957***    -0.3072***
                             (0.0171)      (0.0000)      (0.0177)
         habit_strength     0.8111***     0.6679***     0.8111***
                             (0.0207)      (0.0000)      (0.0187)
              ride_cost    -0.5073***    -0.3159***    -0.5073***
                             (0.0279)      (0.0000)      (0.0239)
   --------------------------------------------------------------
           Observations        72,000        72,000        72,000
         Log-Likelihood    -46,668.43    -46,692.47    -46,668.45
                    AIC      93,342.9      93,390.9      93,342.9
   ==============================================================

NFXP and MCE-IRL achieve nearly identical log-likelihoods (negative 46,668) while CCP is 24 log-likelihood units worse. The Vuong test comparing NFXP and MCE-IRL yields a z-statistic of 1.34 with a p-value of 0.18, confirming statistical equivalence. All three estimators produce similar Brier scores around 0.456.

Counterfactual analysis
-----------------------

The free rides experiment sets the ride cost to zero, simulating a promotional period with no usage fees. This reveals how much ridership would increase if the price barrier were eliminated and how the habit formation mechanism amplifies the initial effect over time.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = nfxp_result.parameters.at[2].set(0.0)
   cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)

Eliminating the ride cost increases expected lifetime welfare by 7.02 utils.

.. list-table:: Ride Cost Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -100%
     - +7.018
     - 0.128
   * - -50%
     - +3.356
     - 0.069
   * - -25%
     - +1.635
     - 0.036
   * - +25%
     - -1.534
     - 0.038
   * - +50%
     - -2.949
     - 0.078
   * - +100%
     - -5.365
     - 0.163

The welfare response is nearly symmetric around the baseline. A 50 percent reduction in ride cost generates a welfare gain of 3.36 utils and shifts average ride probability by 0.069. The policy change column measures the mean absolute difference in choice probabilities across all states, capturing how much the promotion reshapes daily riding decisions.

Running the example
-------------------

.. code-block:: bash

   python examples/citibike-usage/run_estimation.py

Transportation policy interpretation
-------------------------------------

This model captures two key mechanisms in urban transportation behavior. The habit strength parameter measures how past riding reinforces future riding, creating a positive feedback loop that transportation planners can exploit through introductory promotions. The weekend effect captures the systematic difference between commuting (obligatory, regular) and leisure (optional, weather-dependent) riding. The structural model predicts not just the immediate response to a price change but the dynamic equilibrium through the habit channel: a temporary free-ride promotion creates new habits that persist after the promotion ends, generating a long-run ridership increase that exceeds the short-run effect.
