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

Three estimators are fit on 427 training products (80 percent split) over 26 usable periods (11,102 observations). The model uses three features: a holding cost level proportional to inventory, a stockout indicator for low inventory without ordering, and a net promotion effect capturing the combined margin and cost impact of running a promotion.

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
     - NFXP
     - NFXP SE
     - CCP (K=20)
     - MCE-IRL
     - MCE-IRL SE
   * - holding_cost
     - 0.8833
     - 0.0571
     - 0.8948
     - 0.8833
     - 0.0923
   * - stockout_penalty
     - -1.4487
     - 0.0726
     - -1.4417
     - -1.4487
     - 0.1469
   * - net_promotion_effect
     - -0.6028
     - 0.0258
     - -0.6041
     - -0.6029
     - 0.0572

All three parameters are well identified and consistent across estimators. The holding cost is 0.88, significant at the 1 percent level with a standard error of 0.057 under NFXP. The stockout penalty is negative 1.45, roughly 60 percent larger in magnitude than the holding cost. The net promotion effect is negative 0.60, indicating that promotions have a net negative effect on per-period utility after accounting for both the margin reduction and any demand boost. NFXP and MCE-IRL produce identical estimates, while CCP deviates slightly on holding cost (0.89 versus 0.88).

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

All three estimators achieve identical log-likelihoods at negative 14,033 and identical Brier scores at 0.704. The model correctly predicts the retailer's joint decision 32.7 percent of the time against a random baseline of 25 percent for four actions.

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

The welfare response is asymmetric. Reducing the stockout penalty by 50 percent produces a welfare gain of 0.41 utils, while increasing it by 50 percent only costs 0.19 utils. If customers became more tolerant of empty shelves (through substitute availability or online ordering), the retailer could afford to hold less inventory and run promotions more aggressively, capturing the 0.41 utils gain. The small policy changes (0.4 to 1.2 percent) indicate that the retailer already manages inventory conservatively enough to avoid frequent stockouts, so further increasing the stockout penalty has diminishing effect on behavior. This is the signature of a well-optimized retailer operating near the interior of the inaction region.

Running the example
-------------------

.. code-block:: bash

   python examples/supermarket/run_estimation.py

Retail IO interpretation
------------------------

The Aguirregabiria (1999) model demonstrates how dynamic optimization generates pricing patterns that static models cannot explain. A retailer who appears to price erratically in a static cross-section is actually solving an intertemporal problem: running a promotion today depletes inventory, which raises the option value of regular pricing tomorrow. The stockout penalty of negative 1.45 is the dominant cost, roughly 60 percent larger in magnitude than the holding cost of 0.88. This asymmetry drives the retailer to over-order relative to what a myopic model would predict, because the forward-looking retailer internalizes the high cost of empty shelves. The net promotion effect of negative 0.60 means that promotions are costly on net, consistent with trade promotion literature showing that most retail promotions fail to generate positive incremental profit.

Reference
---------

Aguirregabiria, V. (1999). The Dynamics of Markups and Inventories in Retailing Firms. Review of Economic Studies, 66(2), 275-308.
