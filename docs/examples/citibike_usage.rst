Citibike Daily Ride Frequency
==============================

.. image:: /_static/citibike_usage_overview.png
   :alt: Daily Citibike usage patterns showing weekday commuting peaks, weekend leisure riding, and habit formation in ride frequency over time.
   :width: 100%

This example applies structural estimation to a daily ride frequency problem for Citibike members. Each day a member decides whether to take a bikeshare trip. The state captures the day type (weekday or weekend) and recent usage intensity measured as the number of rides in the last seven days. The model recovers preferences over weekend riding, habit strength from recent usage, and the per-ride cost from observed behavior.

The environment has 8 states (2 day types by 4 recent usage buckets: zero rides, one to two rides, three to five rides, and six or more rides in the past week). Riding increases the usage bucket by one and not riding decreases it by one, capturing habit formation and decay. Day type transitions reflect the 5/7 weekday probability in a standard week. Three features enter the ride utility: a weekend indicator, normalized recent usage intensity, and a ride cost indicator.

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

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

Counterfactual analysis
-----------------------

The free rides experiment sets the ride cost to zero, simulating a promotional period with no usage fees. This reveals how much ridership would increase if the price barrier were eliminated and how the habit formation mechanism amplifies the initial effect over time.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy
   new_params = nfxp_result.parameters.at[2].set(0.0)
   cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)

Running the example
-------------------

.. code-block:: bash

   python examples/citibike-usage/run_estimation.py

Transportation policy interpretation
-------------------------------------

This model captures two key mechanisms in urban transportation behavior. The habit strength parameter measures how past riding reinforces future riding, creating a positive feedback loop that transportation planners can exploit through introductory promotions. The weekend effect captures the systematic difference between commuting (obligatory, regular) and leisure (optional, weather-dependent) riding. The structural model predicts not just the immediate response to a price change but the dynamic equilibrium through the habit channel: a temporary free-ride promotion creates new habits that persist after the promotion ends, generating a long-run ridership increase that exceeds the short-run effect.
