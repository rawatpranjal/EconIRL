Citibike Station Destination Choice
=====================================

.. image:: /_static/citibike_route_overview.png
   :alt: NYC Citibike station clusters showing origin-destination flows across Manhattan and Brooklyn with time-of-day variation in trip patterns.
   :width: 100%

This example applies inverse reinforcement learning to Citibike bikeshare trips in New York City. A rider starting at an origin station cluster during a time-of-day window chooses which destination cluster to ride to. The model recovers rider preferences over distance, destination popularity, and peak-hour effects from observed trip data.

The environment has 80 states (20 station clusters from K-Means on geographic coordinates, crossed with 4 time-of-day buckets: night, morning, afternoon, evening). The action space is 20 destination clusters. Three features capture the choice structure: normalized distance between origin and destination centroids, destination cluster popularity measured as the fraction of all trips ending there, and a peak-hour indicator for morning and afternoon periods.

If real Citibike data has not been downloaded, the loader generates synthetic trajectories from the environment with default parameters. To use real data, run the download script first.

Quick start
-----------

.. code-block:: python

   from econirl.environments.citibike_route import CitibikeRouteEnvironment
   from econirl.datasets.citibike_route import load_citibike_route

   env = CitibikeRouteEnvironment(discount_factor=0.95)
   panel = load_citibike_route(as_panel=True)

To download real data first:

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01

Estimation
----------

Two estimators are fit on the route choice data. With 20 destination clusters the action space is large enough that MCE-IRL's occupancy measure computation becomes the binding constraint.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nfxp_result = NFXPEstimator(se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=500)).estimate(panel, utility, env.problem_spec, transitions)

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, mce_result))

Running the example
-------------------

.. code-block:: bash

   python examples/citibike-route/run_estimation.py

Transportation interpretation
-----------------------------

Route choice IRL recovers revealed preferences from observed travel behavior without requiring stated preference surveys. The distance weight captures the disutility of longer trips. The popularity weight reflects agglomeration effects and the attractiveness of well-connected destinations with more docking stations, restaurants, and transit connections. The peak-hour effect captures time pressure during commuting hours when riders may prefer closer destinations. These preference parameters can be used to predict ridership changes from new station installations, rebalancing policies, or pricing changes.
