Counterfactuals Real Bus Engine
================================

This example runs every counterfactual type on the real Rust (1987) bus engine dataset. The true reward function is unknown and may not be linear. The neural estimator (NeuralGLADIUS) is free to learn a different reward surface than the structural model (NFXP). Gaps between the two reflect genuine model flexibility, not estimation error.

The dataset contains 9,410 observations from 90 buses with a replacement rate of 5.1 percent.

Estimation
----------

NFXP recovers theta_c equal to 0.001 and RC equal to 3.07. NeuralGLADIUS trains at beta equal to 0.9999 using the value_scale fix and converges in about 150 epochs.

Reward Heatmap
--------------

The structural reward is a straight line by construction. The neural reward has a different shape. The difference panel reveals where the neural model departs from the linear specification and captures reward nonlinearity.

.. image:: /_static/real_bus_reward_heatmap.png
   :alt: Reward heatmap comparison on real data
   :width: 100%

Policy Comparison
-----------------

The replacement probability curves from both models are correlated at about 0.90 but differ substantially in level. The neural model predicts higher baseline replacement at every state. On real data there is no ground truth to adjudicate which is correct.

.. image:: /_static/real_bus_policy_comparison.png
   :alt: P(replace) comparison on real data
   :width: 100%

Global Perturbation Sweep
-------------------------

Both models respond to a replacement cost penalty in the same direction. The neural model starts from a higher baseline but the gap narrows as the penalty increases and both converge toward zero replacement. The welfare response curves have the same shape but different levels.

.. image:: /_static/real_bus_global_perturbation.png
   :alt: Global perturbation sweep on real data
   :width: 100%

Local Perturbation
------------------

Increasing the operating cost at high-mileage states drives replacement upward in both models. The structural model shows a sharper transition because its lower baseline leaves more room for the effect. Both converge to near-certain replacement at large penalties.

.. image:: /_static/real_bus_local_perturbation.png
   :alt: Local perturbation on real data
   :width: 100%

Transition Counterfactual
-------------------------

Faster depreciation changes the optimal policy for the structural model more than the neural model. The structural welfare drop of 6.56 is much larger than the neural drop of 0.05, reflecting different sensitivities to transition dynamics driven by the different reward surfaces.

.. image:: /_static/real_bus_transition_cf.png
   :alt: Transition counterfactual on real data
   :width: 100%

Choice Set Counterfactual
-------------------------

Mandatory replacement above bin 80 and warranty below bin 10 produce the expected behavior in both models. At forced states the probabilities jump to 0.0 or 1.0. The unconstrained states (10 through 79) show the same qualitative pattern but different levels.

.. image:: /_static/real_bus_choice_set.png
   :alt: Choice set counterfactual on real data
   :width: 100%

Sieve Compression
-----------------

The sieve R-squared of about 0.81 indicates that the linear basis captures most of the neural reward surface but not all of it. The departure from the 45-degree line in the scatter plot reveals where the neural model learned reward patterns that the linear model cannot express. Whether this nonlinearity is real or overfitting depends on the sample size and the neural model's regularization.

.. image:: /_static/real_bus_sieve.png
   :alt: Sieve compression scatter on real data
   :width: 70%

Policy Jacobian
---------------

The Jacobian shows which reward perturbations most affect replacement at each state. The diagonal dominance means own-state rewards matter most. The band structure near the diagonal reflects the transition dynamics where nearby states propagate through the value function.

.. image:: /_static/real_bus_jacobian.png
   :alt: Policy Jacobian heatmap on real data
   :width: 80%

Takeaway
--------

On real data, the neural model finds a reward surface that is 81 percent explained by the linear features. The remaining 19 percent represents potential nonlinearity in the true reward or finite-sample overfitting. Both models agree on the qualitative direction of every counterfactual, but the level differences are larger than on simulated data. Comparing these results to the simulated example (where the truth is linear) helps distinguish genuine nonlinearity from neural estimation noise.

.. code-block:: bash

   python examples/rust-bus-engine/counterfactual_real.py
