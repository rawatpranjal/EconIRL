Core
====

.. module:: econirl.core

The core module provides fundamental types and algorithms for dynamic discrete
choice models.

Types
-----

.. automodule:: econirl.core.types
   :members:
   :undoc-members:
   :show-inheritance:

DDCProblem
~~~~~~~~~~

.. autoclass:: econirl.core.types.DDCProblem
   :members:
   :undoc-members:

Panel
~~~~~

.. autoclass:: econirl.core.types.Panel
   :members:
   :undoc-members:

Trajectory
~~~~~~~~~~

.. autoclass:: econirl.core.types.Trajectory
   :members:
   :undoc-members:

Bellman Operators
-----------------

.. automodule:: econirl.core.bellman
   :members:
   :undoc-members:
   :show-inheritance:

Solvers
-------

.. automodule:: econirl.core.solvers
   :members:
   :undoc-members:
   :show-inheritance:

Feature Specification
---------------------

.. autoclass:: econirl.core.reward_spec.RewardSpec
   :members: compute, compute_gradient, feature_matrix, parameter_names, state_dependent, state_action_dependent, to_linear_utility, to_action_dependent_reward
   :undoc-members:

Data Container
--------------

.. autoclass:: econirl.core.types.TrajectoryPanel
   :members: from_dataframe, sufficient_stats, resample_individuals, iter_transitions, to_dataframe
   :undoc-members:

.. autoclass:: econirl.core.sufficient_stats.SufficientStats
   :members:
   :undoc-members:
