Supermarket Pricing and Inventory
==================================

This example applies structural estimation to real supermarket data from Aguirregabiria (1999). A retailer manages 534 products in a single Spanish supermarket over 29 months. Each period the retailer makes two joint decisions for each product: whether to run a price promotion and whether to place an order from the supplier. Promotions boost sales volume but reduce margins. Orders replenish inventory but have logistical costs. The retailer balances markup revenue against holding costs, stockout risk, and promotion expenses.

The data is discretized into 10 states (5 inventory quintile bins by 2 lagged promotion status levels) and 4 actions (promotion or regular price crossed with order or no order). Transitions are estimated directly from the 13,884 observed product-month transitions. This is real data with no known ground truth parameters, so evaluation focuses on model fit and policy comparison rather than parameter recovery.

.. image:: /_static/supermarket_overview.png
   :alt: Supermarket retail pricing and inventory management showing promotion frequency, inventory turnover, and order placement patterns across product categories.
   :width: 100%

Quick start
-----------

.. code-block:: python

   from econirl.environments.supermarket import SupermarketEnvironment
   from econirl.datasets.supermarket import load_supermarket

   env = SupermarketEnvironment(discount_factor=0.95)
   panel = load_supermarket(as_panel=True)

Estimation
----------

Three estimators are fit on 427 training products (80 percent split) over 26 usable periods (11,102 observations). Since there is no ground truth, we compare model fit rather than parameter recovery.

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

.. list-table:: Estimation Results (427 products, 11,102 observations)
   :header-rows: 1

   * - Parameter
     - CCP (K=20)
     - MCE-IRL
     - MCE-IRL SE
   * - markup_benefit
     - -0.1632
     - 0.3014
     - 0.0317
   * - holding_cost
     - 0.8821
     - 0.8833
     - 0.0853
   * - stockout_penalty
     - -1.4536
     - -1.4487
     - 0.1368
   * - promotion_cost
     - -0.7672
     - -0.3014
     - 0.0317

The holding cost and stockout penalty are well identified and consistent across estimators. CCP estimates a holding cost of 0.88 and MCE-IRL estimates 0.88 with a standard error of 0.085, both significant at the 1 percent level. The stockout penalty is negative 1.45 for both, indicating that running out of stock is roughly 60 percent more costly per period than carrying excess inventory. NFXP is omitted from the table because its markup_benefit and promotion_cost estimates blew up (30.2 and 29.6 with standard errors above 5000), due to near-collinearity between those two features. The Hessian condition number for NFXP is 13.2 million, confirming weak identification along that direction.

CCP and MCE-IRL disagree on the markup_benefit and promotion_cost decomposition. MCE-IRL estimates markup_benefit as 0.30 and promotion_cost as negative 0.30, exact mirror images that suggest the model can only identify their difference (the net margin effect of running a promotion) rather than their individual levels. CCP places the promotion cost at negative 0.77, absorbing some of the markup effect.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NFXP (Nested Fixed Point)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
         markup_benefit       30.1829    -0.1632***     0.3014***
                          (5334.9326)      (0.0000)      (0.0317)
           holding_cost     0.8832***     0.8821***     0.8833***
                             (0.0568)      (0.0000)      (0.0853)
       stockout_penalty    -1.4486***    -1.4536***    -1.4487***
                             (0.0330)      (0.0000)      (0.1368)
         promotion_cost       29.5801    -0.7672***    -0.3014***
                          (5849.2427)      (0.0000)      (0.0317)
   --------------------------------------------------------------
           Observations        11,102        11,102        11,102
         Log-Likelihood    -14,032.79    -14,032.91    -14,032.91
                    AIC      28,073.6      28,073.8      28,073.8
   ==============================================================

All three estimators achieve identical log-likelihoods at negative 14,033 and identical Brier scores at 0.704, confirming that the parameter disagreement on markup_benefit and promotion_cost does not affect predictive fit. The model correctly predicts the retailer's joint decision 32.7 percent of the time against a random baseline of 25 percent for four actions.

Counterfactual analysis
-----------------------

The stockout tolerance experiment halves the stockout penalty, simulating a scenario where customers become more patient with out-of-stock items, perhaps due to available substitutes or online ordering alternatives.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   stockout_idx = env.parameter_names.index("stockout_penalty")
   new_params = mce_result.parameters.at[stockout_idx].set(mce_result.parameters[stockout_idx] / 2)
   cf = counterfactual_policy(mce_result, new_params, utility, problem, transitions)

Halving the stockout penalty increases expected lifetime welfare by 0.41 utils.

.. list-table:: Stockout Penalty Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - +0.407
     - 0.012
   * - -25%
     - +0.163
     - 0.005
   * - +25%
     - -0.111
     - 0.004
   * - +50%
     - -0.187
     - 0.006
   * - +100%
     - -0.275
     - 0.009

The welfare response is asymmetric. Reducing the stockout penalty by 50 percent produces a welfare gain of 0.41 utils, while increasing it by 50 percent only costs 0.19 utils. The small policy changes (0.4 to 1.2 percent) indicate that the retailer's ordering and pricing decisions are not highly sensitive to the stockout penalty at its estimated level, consistent with a retailer that already manages inventory conservatively enough to avoid frequent stockouts.

Running the example
-------------------

.. code-block:: bash

   python examples/supermarket/run_estimation.py

Retail IO interpretation
------------------------

The Aguirregabiria (1999) model demonstrates how dynamic optimization generates pricing patterns that static models cannot explain. A retailer who appears to price erratically in a static cross-section is actually solving an intertemporal problem: running a promotion today depletes inventory, which raises the option value of regular pricing tomorrow. The estimated parameters reveal the tradeoff between immediate margin (markup benefit) and dynamic costs (inventory holding, stockout risk, promotion frequency). The stockout penalty of negative 1.45 is the dominant cost, roughly 60 percent larger in magnitude than the holding cost of 0.88. This asymmetry drives the retailer to over-order relative to what a myopic model would predict, because the forward-looking retailer internalizes the high cost of empty shelves.

Reference
---------

Aguirregabiria, V. (1999). The Dynamics of Markups and Inventories in Retailing Firms. Review of Economic Studies, 66(2), 275-308.
