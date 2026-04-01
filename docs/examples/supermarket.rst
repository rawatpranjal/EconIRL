Supermarket Pricing and Inventory
==================================

.. image:: /_static/supermarket_overview.png
   :alt: Supermarket retail pricing and inventory management showing promotion frequency, inventory turnover, and order placement patterns across product categories.
   :width: 100%

This example applies structural estimation to real supermarket data from Aguirregabiria (1999). A retailer manages 534 products in a single Spanish supermarket over 29 months. Each period the retailer makes two joint decisions for each product: whether to run a price promotion and whether to place an order from the supplier. Promotions boost sales volume but reduce margins. Orders replenish inventory but have logistical costs. The retailer balances markup revenue against holding costs, stockout risk, and promotion expenses.

The data is discretized into 10 states (5 inventory quintile bins by 2 lagged promotion status levels) and 4 actions (promotion or regular price crossed with order or no order). Transitions are estimated directly from the 13,884 observed product-month transitions. This is real data with no known ground truth parameters, so evaluation focuses on model fit and policy comparison rather than parameter recovery.

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

Post-estimation diagnostics
---------------------------

The ``etable`` function places all three models side by side with significance stars.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

Running the example
-------------------

.. code-block:: bash

   python examples/supermarket/run_estimation.py

Retail IO interpretation
------------------------

The Aguirregabiria (1999) model demonstrates how dynamic optimization generates pricing patterns that static models cannot explain. A retailer who appears to price erratically in a static cross-section is actually solving an intertemporal problem: running a promotion today depletes inventory, which raises the option value of regular pricing tomorrow. The estimated parameters reveal the tradeoff between immediate margin (markup benefit) and dynamic costs (inventory holding, stockout risk, promotion frequency). This is a canonical example of how DDC estimation recovers the structural primitives behind seemingly complex pricing behavior.
