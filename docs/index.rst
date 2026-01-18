econirl: The StatsModels of IRL
================================

**econirl** bridges Structural Econometrics and Inverse Reinforcement Learning,
providing economist-friendly APIs for estimating dynamic discrete choice models.

.. code-block:: python

   from econirl import RustBusEnvironment, LinearUtility, NFXPEstimator
   from econirl.simulation import simulate_panel

   # Set up environment and simulate data
   env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)
   panel = simulate_panel(env, n_individuals=500, n_periods=100)

   # Estimate parameters
   utility = LinearUtility.from_environment(env)
   result = NFXPEstimator().estimate(panel, utility, env.problem_spec, env.transition_matrices)

   # StatsModels-style output
   print(result.summary())

Key Features
------------

**Economist-Friendly API**
   Familiar terminology: utility, preferences, characteristics.
   StatsModels-style ``summary()`` with standard errors and hypothesis tests.

**Multiple Estimation Methods**
   NFXP (Nested Fixed Point), CCP estimators, and MaxEnt IRL.
   Choose the right method for your application.

**Gymnasium Compatible**
   Environments follow the Gymnasium API.
   Seamlessly integrate with RL tooling.

**Rich Inference**
   Standard errors, confidence intervals, hypothesis tests.
   Counterfactual analysis and welfare calculations.


Getting Started
---------------

.. toctree::
   :maxdepth: 1

   installation
   quickstart

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorials/index

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
