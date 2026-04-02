Citibike Station Destination Choice
=====================================

This example applies inverse reinforcement learning to real Citibike bikeshare trips from January 2024. A rider starting at an origin station cluster during a time-of-day window chooses which destination cluster to ride to. The model recovers rider preferences over distance, destination popularity, and peak-hour effects from 1.88 million observed trips.

The environment has 80 states (20 station clusters from K-Means on geographic coordinates, crossed with 4 time-of-day buckets: night, morning, afternoon, evening). The action space is 20 destination clusters. Three features capture the choice structure: normalized distance between origin and destination centroids, destination cluster popularity measured as the fraction of all trips ending there, and a peak-hour indicator for morning and afternoon periods.

.. image:: /_static/citibike_route_overview.png
   :alt: NYC Citibike station clusters showing origin-destination flows across Manhattan and Brooklyn with time-of-day variation in trip patterns.
   :width: 100%

Quick start
-----------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01

.. code-block:: python

   from econirl.environments.citibike_route import CitibikeRouteEnvironment
   from econirl.datasets.citibike_route import load_citibike_route

   env = CitibikeRouteEnvironment(discount_factor=0.95)
   panel = load_citibike_route(as_panel=True)

Estimation
----------

Three estimators are fit on 799 training riders (39,950 trip observations) from the January 2024 Citibike system data.

.. code-block:: python

   from econirl.estimation.nnes import NNESEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nnes_result = NNESEstimator(se_method="asymptotic").estimate(panel, utility, env.problem_spec, transitions)
   ccp_result = CCPEstimator(num_policy_iterations=20, se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=500)).estimate(panel, utility, env.problem_spec, transitions)

.. list-table:: Estimation Results (799 riders, 39,950 observations)
   :header-rows: 1

   * - Parameter
     - NNES
     - CCP (K=20)
     - MCE-IRL
     - MCE-IRL SE
   * - distance_weight
     - -4.0550
     - -4.0406
     - -4.0339
     - 0.0429
   * - popularity_weight
     - -0.0199
     - 0.7110
     - 0.0000
     - 0.0000
   * - peak_weight
     - 0.0100
     - -0.8636
     - 0.0000
     - 0.0003

All three estimators agree on the distance weight at approximately negative 4.0, and this parameter is statistically significant with a standard error of 0.043 under MCE-IRL. The distance weight is four times larger than the synthetic environment's default of negative 1.0, reflecting the strong preference of real NYC riders for short trips in January. CCP and MCE-IRL achieve nearly identical log-likelihoods (negative 104,325) while NNES is 700 units worse. Prediction accuracy is 60 percent against a random baseline of 5 percent for 20 destinations, indicating strong concentration of trips toward nearby clusters.

The popularity and peak features remain poorly identified across all estimators. CCP finds a positive popularity weight (0.71) and a negative peak weight (negative 0.86), but MCE-IRL and NNES estimate both near zero. The fundamental identification issue is that distance dominates destination choice so strongly in real data that the remaining features have insufficient residual variation to pin down their effects.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nnes_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NNES-NFXP (Bellman residual)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
        distance_weight       -4.0550    -4.0406***    -4.0339***
                                (nan)      (0.0000)      (0.0429)
      popularity_weight       -0.0199     0.7110***        0.0000
                                (nan)      (0.0000)      (0.0000)
            peak_weight        0.0100    -0.8636***        0.0000
                                (nan)      (0.0000)      (0.0003)
   --------------------------------------------------------------
           Observations        39,950        39,950        39,950
         Log-Likelihood   -105,024.38   -104,324.82   -104,324.89
                    AIC     210,054.8     208,655.6     208,655.8
   ==============================================================

Counterfactual analysis
-----------------------

The infrastructure improvement experiment halves the distance disutility, simulating a scenario where protected bike lanes and e-bikes make longer trips less costly.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = mce_result.parameters.at[0].set(mce_result.parameters[0] / 2)
   cf = counterfactual_policy(mce_result, new_params, utility, problem, transitions)

Halving the distance disutility increases expected lifetime welfare by 11.0 utils.

.. list-table:: Distance Weight Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - +10.966
     - 0.016
   * - -25%
     - +5.037
     - 0.008
   * - +25%
     - -4.268
     - 0.007
   * - +50%
     - -7.862
     - 0.014
   * - +100%
     - -13.357
     - 0.025

The welfare response is nearly linear and mildly asymmetric. Reducing the distance disutility by 50 percent gains 11.0 utils while increasing it by 50 percent costs 7.9 utils. The small average policy changes (0.7 to 2.5 percent) indicate that most trips already go to nearby clusters, so reducing distance costs mainly benefits the marginal long-distance trips rather than restructuring the overall destination distribution.

Running the example
-------------------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01
   python examples/citibike-route/run_estimation.py

Transportation interpretation
-----------------------------

Route choice IRL recovers revealed preferences from observed travel behavior without requiring stated preference surveys. The distance weight of negative 4.0 is the dominant parameter, reflecting the strong aversion to long trips in winter conditions. This magnitude is consistent with the well-documented tendency of bikeshare riders to prefer trips under 15 minutes. The weak identification of popularity and peak features in real data suggests that those dimensions do not vary enough across the 20 destination clusters to be separately identified from distance, or that riders' actual choice sets are more constrained than the model assumes. A richer feature set with station-level amenity counts, elevation profiles, or transit connectivity scores might improve identification of non-distance factors.
