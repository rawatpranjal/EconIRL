Shanghai Route Choice
=====================

This example applies five estimators to taxi route-choice data from Shanghai (Zhao and Liang 2022). Drivers choose which road segment to traverse at each intersection, trading off road length against road type. The dataset has 714 road segments, 8 compass directions, and 10,000 observed routes.

.. code-block:: python

   from econirl.datasets.shanghai_route import (
       load_shanghai_network, load_shanghai_trajectories,
       parse_trajectories_to_panel, build_transition_matrix,
       build_edge_features, build_state_action_features,
   )

   network = load_shanghai_network()
   train_df = load_shanghai_trajectories(split="train", cv=0, size=1000)
   panel = parse_trajectories_to_panel(train_df, network["transit"])
   transitions = build_transition_matrix(network["transit"])
   features = build_state_action_features(
       build_edge_features(network["edges"]), network["transit"]
   )

The features describe each road segment. The length feature captures how long the segment is. The remaining six features are indicator variables for the road type.

.. list-table:: Benchmark Results (1000 training routes, 4893 test routes)
   :header-rows: 1

   * - Estimator
     - Test LL per step
     - Step Accuracy
     - Time
   * - BC (empirical)
     - -0.54
     - 84.2%
     - 0s
   * - NFXP
     - -0.74
     - 65.0%
     - 6s
   * - CCP
     - -0.69
     - 77.8%
     - 3s
   * - NNES
     - -0.63
     - 69.2%
     - 35s
   * - TD-CCP
     - -0.88
     - 67.3%
     - 22s
   * - NeuralGLADIUS
     - -0.90
     - 73.8%
     - 27s

Behavioral cloning wins on pure prediction because 714 states with 18,000 training transitions gives enough data to memorize the empirical choice probabilities. The structural estimators sacrifice in-sample fit for interpretable parameters. All three structural methods agree that drivers prefer shorter road segments and avoid residential streets in favor of primary roads. NeuralGLADIUS uses destination context but its projection R-squared of 0.06 shows that seven linear features capture only 6 percent of the learned neural reward. The remaining 94 percent is nonlinear interaction effects that a linear model cannot represent.
