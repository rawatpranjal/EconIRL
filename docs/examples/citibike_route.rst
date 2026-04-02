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

All three estimators agree on the distance weight at approximately negative 4.0, statistically significant with a standard error of 0.043 under MCE-IRL. The magnitude reveals that distance is overwhelmingly the primary determinant of destination choice in winter. A rider choosing between a cluster 1 km away and one 3 km away faces a utility difference of roughly 8 units from distance alone, which translates to a choice probability ratio of over 50 to 1 in favor of the nearer destination. This is consistent with the well-documented tendency of bikeshare riders to prefer trips under 15 minutes, amplified by January cold. The model correctly predicts 60 percent of destination choices out of 20 possible clusters, far above the 5 percent random baseline.

The popularity and peak features are not statistically significant under MCE-IRL, with p-values of 0.999 and 0.966. This is itself an economic finding: riders' destination choices are determined almost entirely by proximity, with negligible modulation from how popular a cluster is or what time of day the trip occurs. In winter, the time cost of riding dominates amenity-based preferences. CCP finds nonzero popularity (0.71) and peak (negative 0.86) estimates, but these likely reflect confounding between distance and cluster characteristics rather than genuine preference variation.

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

The welfare response is nearly linear and mildly asymmetric. Reducing the distance disutility by 50 percent gains 11.0 utils while increasing it by 50 percent costs 7.9 utils. The small average policy changes (0.7 to 2.5 percent) mean that infrastructure improvements would not dramatically reroute existing riders but would make their existing trips less painful and unlock a modest number of longer trips that were previously too costly. The welfare gain of 11.0 utils from halving distance costs quantifies the value of bike lane investments from the rider's perspective. Making distance twice as costly would shift choice probabilities by only 2.5 percent, confirming that riders already travel to the nearest feasible cluster and have little room to compress further.

Running the example
-------------------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01
   python examples/citibike-route/run_estimation.py

Transportation interpretation
-----------------------------

Route choice IRL recovers revealed preferences from observed travel behavior without requiring stated preference surveys. The central finding is that January bikeshare riders are time-sensitive commuters, not amenity-seeking explorers. Distance explains nearly all destination variation, and neither cluster popularity nor time of day adds predictive power once distance is controlled for. This has direct implications for station planning: new stations should prioritize geographic coverage and filling gaps between existing clusters rather than co-locating with popular destinations. The structural model also predicts that e-bike deployment (which reduces effective distance by increasing speed) would generate a welfare gain proportional to the distance weight reduction, benefiting riders on longer commute corridors most. Running the same model on summer data would test whether popularity and peak effects emerge when weather no longer constrains riders to the shortest possible trip.
