Counterfactuals Simulated Bus Engine
=====================================

This example runs every counterfactual type on simulated bus engine data where the ground truth reward is linear. Both the structural estimator (NFXP) and the neural estimator (NeuralGLADIUS) are fit on the same data, and every counterfactual is computed on both reward surfaces side by side. Because the true reward IS linear, the neural model should recover the same counterfactual predictions as the structural model. Any gap that remains is estimation error from the neural network, not model misspecification.

The data come from ``RustBusEnvironment`` with true parameters theta_c equal to 0.001 and RC equal to 3.0. We simulate 200 buses over 100 periods, giving 20,000 observations.

Estimation
----------

NFXP recovers theta_c equal to 0.0012 and RC equal to 3.01, close to the ground truth. NeuralGLADIUS trains for about 160 epochs with the value_scale fix that enables training at beta equal to 0.9999.

Reward Heatmap
--------------

The first comparison shows the learned reward surfaces side by side. The structural reward is linear by construction. The neural reward has a similar gradient but different absolute levels. The difference panel reveals where the two diverge.

.. image:: /_static/sim_bus_reward_heatmap.png
   :alt: Reward heatmap comparison on simulated data
   :width: 100%

Policy Comparison
-----------------

The structural and neural models produce replacement probability curves that are highly correlated (above 0.9) but differ in level. The true policy from the data-generating process is shown as a dashed reference. NFXP tracks the truth closely. The neural model overestimates replacement probability at all states.

.. image:: /_static/sim_bus_policy_comparison.png
   :alt: P(replace) comparison on simulated data
   :width: 100%

Global Perturbation Sweep
-------------------------

Adding a uniform penalty to the replacement action traces out how each model's policy responds to a cost increase. Both models decrease replacement probability monotonically as the penalty grows. The gap between them narrows at large penalties because both converge toward zero replacement. The direction of the response is identical.

.. image:: /_static/sim_bus_global_perturbation.png
   :alt: Global perturbation sweep on simulated data
   :width: 100%

Local Perturbation
------------------

Increasing the operating cost only at high-mileage states (bins above 60) pushes replacement probability upward at those states. Both models respond in the same direction. The neural model starts from a higher baseline and saturates faster.

.. image:: /_static/sim_bus_local_perturbation.png
   :alt: Local perturbation on simulated data
   :width: 100%

Transition Counterfactual
-------------------------

Changing the mileage increment distribution from (0.39, 0.60, 0.01) to (0.20, 0.50, 0.30) simulates faster engine depreciation. Both models show slightly increased replacement at high mileage under the new transitions. The welfare drop is larger for the structural model because its value function is more sensitive to transition changes at high beta.

.. image:: /_static/sim_bus_transition_cf.png
   :alt: Transition counterfactual on simulated data
   :width: 100%

Choice Set Counterfactual
-------------------------

Mandatory replacement above mileage bin 80 and a warranty preventing replacement below bin 10 constrain the action set. Both models agree exactly at forced states (probability 0.0 in the warranty zone, 1.0 in the mandatory zone). The gap at unconstrained states (10 through 79) reflects the baseline level difference, not a disagreement about the constraint effect.

.. image:: /_static/sim_bus_choice_set.png
   :alt: Choice set counterfactual on simulated data
   :width: 100%

Sieve Compression
-----------------

Projecting the neural reward onto the same linear features used by NFXP tests how much of the neural surface is captured by the linear specification. The R-squared of about 0.78 indicates the neural model has learned a reward that is mostly but not entirely linear. The scatter plot shows the neural reward differences against the structural reward differences with a 45-degree reference line.

.. image:: /_static/sim_bus_sieve.png
   :alt: Sieve compression scatter on simulated data
   :width: 70%

Policy Jacobian
---------------

The Jacobian heatmap shows how perturbing the replacement reward at state s-prime affects the replacement probability at state s. The strongest effects are along the diagonal (own-state perturbations) and at low mileage states (which affect the continuation value for all higher states through the transition dynamics).

.. image:: /_static/sim_bus_jacobian.png
   :alt: Policy Jacobian heatmap on simulated data
   :width: 80%

Takeaway
--------

On simulated linear data, the neural model agrees with the structural model on the direction of every counterfactual but differs in level. The sieve R-squared of 0.78 confirms that the neural reward is not perfectly linear, reflecting estimation difficulty rather than genuine nonlinearity. Structural estimation remains more efficient when the specification is correct.

.. code-block:: bash

   python examples/rust-bus-engine/counterfactual_sim.py
