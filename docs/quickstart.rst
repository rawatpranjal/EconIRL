Quickstart
==========

This guide walks through a complete estimation workflow using the classic
Rust (1987) bus engine replacement model.

The Problem
-----------

Harold Zurcher manages a fleet of buses and must decide when to replace engines.
Each period, he observes the mileage and chooses to either:

- **Keep** the current engine (action 0)
- **Replace** the engine (action 1)

The utility functions are:

- Keep: :math:`u(s, \text{keep}) = -\theta_c \cdot \text{mileage}(s) + \varepsilon_0`
- Replace: :math:`u(s, \text{replace}) = -RC + \varepsilon_1`

where :math:`\theta_c` is the operating cost parameter, :math:`RC` is the
replacement cost, and :math:`\varepsilon` are i.i.d. Type I Extreme Value shocks.

Step 1: Create the Environment
------------------------------

.. code-block:: python

   from econirl import RustBusEnvironment

   # Create environment with true parameters
   env = RustBusEnvironment(
       operating_cost=0.001,    # Cost per mileage unit
       replacement_cost=3.0,    # Fixed replacement cost
       num_mileage_bins=90,     # State space discretization
       discount_factor=0.9999,  # High discount factor (patient agent)
   )

   print(env.describe())

Step 2: Simulate Panel Data
---------------------------

Generate synthetic data from the model:

.. code-block:: python

   from econirl.simulation import simulate_panel

   panel = simulate_panel(
       env,
       n_individuals=500,   # Number of buses
       n_periods=100,       # Periods per bus
       seed=42,
   )

   print(f"Observations: {panel.num_observations:,}")
   print(f"Replacement rate: {panel.get_all_actions().float().mean():.2%}")

Step 3: Set Up Estimation
-------------------------

Define the utility specification and create an estimator:

.. code-block:: python

   from econirl import LinearUtility, NFXPEstimator

   # Utility specification (matches environment structure)
   utility = LinearUtility.from_environment(env)

   # NFXP estimator with asymptotic standard errors
   estimator = NFXPEstimator(
       se_method="asymptotic",
       verbose=True,
   )

Step 4: Estimate Parameters
---------------------------

Run the estimation:

.. code-block:: python

   result = estimator.estimate(
       panel=panel,
       utility=utility,
       problem=env.problem_spec,
       transitions=env.transition_matrices,
   )

Step 5: View Results
--------------------

Get StatsModels-style output:

.. code-block:: python

   print(result.summary())

Output::

   ================================================================================
                      Dynamic Discrete Choice Estimation Results
   ================================================================================
   Method:                    NFXP (Nested Fixed Point)
   No. Observations:          50,000
   Log-Likelihood:            -9,500.37
   --------------------------------------------------------------------------------
                              coef    std err        t    P>|t|   [0.025   0.975]
   --------------------------------------------------------------------------------
   operating_cost           0.0010     0.0001    10.23    0.000   0.0008   0.0012
   replacement_cost         2.9854     0.0523    57.12    0.000   2.8829   3.0879
   --------------------------------------------------------------------------------

Step 6: Counterfactual Analysis
-------------------------------

Analyze policy changes under different scenarios:

.. code-block:: python

   from econirl.simulation import counterfactual_policy

   # What if replacement cost increased 50%?
   new_params = result.parameters.clone()
   new_params[1] *= 1.5

   cf = counterfactual_policy(
       result=result,
       new_parameters=new_params,
       utility=utility,
       problem=env.problem_spec,
       transitions=env.transition_matrices,
   )

   print(f"Baseline avg replacement rate: {cf.baseline_policy[:, 1].mean():.4f}")
   print(f"Counterfactual avg replacement rate: {cf.counterfactual_policy[:, 1].mean():.4f}")

Next Steps
----------

- See :doc:`tutorials/index` for in-depth examples
- Explore the :doc:`api/index` for detailed documentation
