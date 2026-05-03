Environments
============

.. module:: econirl.environments

Environments define the structure of dynamic discrete choice problems:
state spaces, action spaces, transition dynamics, and reward structures.

All environments follow the `Gymnasium <https://gymnasium.farama.org/>`_ API,
making them compatible with standard RL tooling.

RustBusEnvironment
------------------

.. autoclass:: econirl.environments.rust_bus.RustBusEnvironment
   :members:
   :undoc-members:
   :show-inheritance:

   The classic Rust (1987) bus engine replacement environment.

   **Example:**

   .. code-block:: python

      from econirl import RustBusEnvironment

      env = RustBusEnvironment(
          operating_cost=0.001,
          replacement_cost=3.0,
          num_mileage_bins=90,
          discount_factor=0.9999,
      )

      print(env.describe())

Base Classes
------------

.. automodule:: econirl.environments.base
   :members:
   :undoc-members:
   :show-inheritance:

Creating Custom Environments
----------------------------

To create a custom environment, subclass ``DDCEnvironment`` and implement:

1. ``__init__``: Define state/action spaces and parameters
2. ``step``: State transition logic
3. ``reset``: Initialize episode
4. ``describe``: Human-readable description
5. ``problem_spec``: Return ``DDCProblem`` instance
6. ``transition_matrices``: Return transition probability tensors

**Example:**

.. code-block:: python

   from econirl.environments.base import DDCEnvironment
   from econirl.core.types import DDCProblem
   import torch

   class MyEnvironment(DDCEnvironment):
       def __init__(self, param1, param2, **kwargs):
           super().__init__(**kwargs)
           self.param1 = param1
           self.param2 = param2
           # Define spaces, transitions, etc.

       @property
       def problem_spec(self) -> DDCProblem:
           return DDCProblem(
               num_states=self.num_states,
               num_actions=self.num_actions,
               discount_factor=self.discount_factor,
               scale_parameter=self.scale_parameter,
           )
