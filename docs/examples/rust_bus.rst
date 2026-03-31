Rust Bus Engine Replacement
===========================

This example replicates Rust (1987) on the bus engine replacement dataset. A fleet manager decides each month whether to keep running a bus engine or replace it. The operating cost rises with mileage and the replacement cost is fixed.

The state space has 90 mileage bins and 2 actions. All four structural estimators recover the same operating cost and replacement cost parameters.

.. code-block:: python

   from econirl import NFXP, CCP, NNES, TDCCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp  = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp   = CCP(discount=0.9999, num_policy_iterations=5).fit(df, state="mileage_bin", action="replaced", id="bus_id")

   print(nfxp.params_)  # {'theta_c': 0.001, 'RC': 3.07}
   print(ccp.params_)   # {'theta_c': 0.001, 'RC': 3.07}

.. list-table:: Parameter Recovery (200 individuals, 100 periods, simulated)
   :header-rows: 1

   * - Estimator
     - theta_c
     - RC
     - LL
     - Time
   * - NFXP
     - 0.0012
     - 3.011
     - -4263
     - 0.1s
   * - CCP (K=20)
     - 0.0012
     - 3.011
     - -4263
     - 0.1s
   * - MCE-IRL
     - 0.0012
     - 3.011
     - -4263
     - 0.1s
   * - NNES
     - 0.0315
     - 3.073
     - -4264
     - 16s
   * - TD-CCP
     - 0.0011
     - 2.943
     - -4265
     - 130s

NFXP, CCP, and MCE-IRL recover identical parameters. CCP avoids the inner Bellman loop entirely and matches NFXP in a fraction of the time. MCE-IRL reaches the same answer from the IRL side, confirming the theoretical equivalence between maximum causal entropy IRL and logit DDC estimation. The neural estimators (NNES and TD-CCP) get close but introduce small approximation error from the value network. This is expected. Neural methods are designed for high-dimensional problems where exact methods cannot run.

Counterfactual analysis
-----------------------

Structural estimation enables counterfactual policy simulation. Once we have estimated the operating cost and replacement cost parameters, we can ask what happens to replacement behavior under different economic conditions.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy

   # What if replacement cost doubles?
   new_params = result.parameters.clone()
   new_params[1] *= 2  # replacement_cost
   cf = counterfactual_policy(result, new_params, utility, problem, transitions)
   print(cf.policy[50, :])  # P(keep|mileage=50), P(replace|mileage=50)

.. image:: /_static/rust_bus_counterfactual.png
   :alt: Counterfactual analysis showing replacement probability and relative value function under varying replacement cost and operating cost.
   :width: 100%

The top row shows how replacement probability changes across mileage levels under two types of parameter variation. The top-left panel varies the replacement cost RC. Halving the replacement cost to 1.5 raises the replacement probability at every mileage level because the manager can afford to replace engines more freely. Tripling it to 9.0 suppresses replacement almost entirely, forcing the manager to run engines at high mileage. The top-right panel varies the per-mile operating cost. Doubling the operating cost makes high-mileage operation expensive, so the manager replaces earlier. Halving it makes running the engine cheap, so the manager holds off on replacement.

The bottom row shows the relative value function V(s) minus V(0) under each scenario. Plotting relative to mileage bin zero reveals the shape that is hidden when looking at the raw value function, where the absolute level with discount factor 0.9999 dominates the scale. The value function declines with mileage in every scenario because higher mileage means higher cumulative operating costs ahead. The decline is steepest when the replacement cost is high or the operating cost is high, because the manager faces larger losses from running a worn engine and has fewer attractive exit options.
