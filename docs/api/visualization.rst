Visualization
=============

.. module:: econirl.visualization

Plotting utilities for policies, value functions, and estimation results.

Policy Visualization
--------------------

.. automodule:: econirl.visualization.policy
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from econirl.visualization.policy import plot_policy

   # After estimation
   plot_policy(
       result.policy,
       title="Estimated Replacement Policy",
       xlabel="Mileage Bin",
       ylabel="P(Replace)",
   )

Value Function Visualization
----------------------------

.. automodule:: econirl.visualization.value
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from econirl.visualization.value import plot_value_function, plot_q_values

   # Plot V(s)
   plot_value_function(V, title="Value Function")

   # Plot Q(s, a) for each action
   plot_q_values(Q, action_names=["Keep", "Replace"])

Common Plotting Patterns
------------------------

**Comparing Policies:**

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   ax.plot(baseline_policy[:, 1], label="Baseline")
   ax.plot(counterfactual_policy[:, 1], label="Counterfactual")
   ax.set_xlabel("State")
   ax.set_ylabel("P(Action 1)")
   ax.legend()
   plt.show()

**Model Fit:**

.. code-block:: python

   # Compare predicted vs empirical
   fig, ax = plt.subplots()

   # Predicted
   ax.plot(states, predicted_policy[:, 1], 'b-', label="Model")

   # Empirical
   ax.scatter(states, empirical_rates, alpha=0.5, label="Data")

   ax.legend()
   plt.show()

**Parameter Recovery:**

.. code-block:: python

   # Visualize estimate with confidence interval
   fig, ax = plt.subplots()

   ci_low, ci_high = result.confidence_interval()
   ax.errorbar(
       0, result.parameters[0],
       yerr=[[result.parameters[0] - ci_low[0]], [ci_high[0] - result.parameters[0]]],
       fmt='o', capsize=5
   )
   ax.axhline(true_value, color='red', linestyle='--', label="True")
   ax.legend()
   plt.show()
