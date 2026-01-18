Simulation
==========

.. module:: econirl.simulation

Tools for simulating data from dynamic discrete choice models and performing
counterfactual analysis.

Panel Simulation
----------------

.. automodule:: econirl.simulation.synthetic
   :members:
   :undoc-members:
   :show-inheritance:

simulate_panel
~~~~~~~~~~~~~~

.. autofunction:: econirl.simulation.synthetic.simulate_panel

   Generate synthetic panel data from a DDC environment.

   **Example:**

   .. code-block:: python

      from econirl import RustBusEnvironment
      from econirl.simulation import simulate_panel

      env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)

      panel = simulate_panel(
          env,
          n_individuals=500,
          n_periods=100,
          seed=42,
      )

      print(f"Observations: {panel.num_observations}")
      print(f"Replacement rate: {panel.get_all_actions().float().mean():.2%}")

Counterfactual Analysis
-----------------------

.. automodule:: econirl.simulation.counterfactual
   :members:
   :undoc-members:
   :show-inheritance:

counterfactual_policy
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: econirl.simulation.counterfactual.counterfactual_policy

   Compute policy under alternative parameters.

   **Example:**

   .. code-block:: python

      from econirl.simulation import counterfactual_policy

      # After estimation
      new_params = result.parameters.clone()
      new_params[1] *= 1.5  # 50% increase in replacement cost

      cf = counterfactual_policy(
          result=result,
          new_parameters=new_params,
          utility=utility,
          problem=problem,
          transitions=transitions,
      )

      print(f"Baseline replacement rate: {cf.baseline_policy[:, 1].mean():.4f}")
      print(f"Counterfactual rate: {cf.counterfactual_policy[:, 1].mean():.4f}")

compute_welfare_effect
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: econirl.simulation.counterfactual.compute_welfare_effect

   Compute welfare effects of policy changes.

   **Example:**

   .. code-block:: python

      from econirl.simulation import compute_welfare_effect

      welfare = compute_welfare_effect(cf, transitions, use_stationary=True)

      print(f"Baseline welfare: {welfare['baseline_expected_value']:.4f}")
      print(f"Counterfactual welfare: {welfare['counterfactual_expected_value']:.4f}")
      print(f"Welfare change: {welfare['total_welfare_change']:.4f}")
