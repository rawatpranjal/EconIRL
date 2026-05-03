Preferences
===========

.. module:: econirl.preferences

Preferences (utility specifications) define how state-action features map to
utility values. The utility function determines agent behavior in the model.

LinearUtility
-------------

.. autoclass:: econirl.preferences.linear.LinearUtility
   :members:
   :undoc-members:
   :show-inheritance:

   Linear-in-parameters utility specification.

   **Example:**

   .. code-block:: python

      from econirl import LinearUtility, RustBusEnvironment

      # Create from environment
      env = RustBusEnvironment()
      utility = LinearUtility.from_environment(env)

      # Or create manually
      import torch
      feature_matrix = torch.randn(90, 2, 2)  # (states, actions, features)
      utility = LinearUtility(
          feature_matrix=feature_matrix,
          parameter_names=['cost1', 'cost2']
      )

      # Compute utility matrix for given parameters
      params = torch.tensor([0.001, 3.0])
      U = utility.compute(params)  # Shape: (states, actions)

Base Classes
------------

.. automodule:: econirl.preferences.base
   :members:
   :undoc-members:
   :show-inheritance:

Custom Utility Specifications
-----------------------------

To create custom utility specifications, subclass ``UtilitySpecification``:

.. code-block:: python

   from econirl.preferences.base import UtilitySpecification
   import torch

   class NonlinearUtility(UtilitySpecification):
       def __init__(self, num_states, num_actions):
           self.num_states = num_states
           self.num_actions = num_actions

       @property
       def num_parameters(self) -> int:
           return 3  # Example: 3 parameters

       @property
       def parameter_names(self) -> list[str]:
           return ['alpha', 'beta', 'gamma']

       def compute(self, parameters: torch.Tensor) -> torch.Tensor:
           # Return utility matrix of shape (num_states, num_actions)
           alpha, beta, gamma = parameters
           # ... compute nonlinear utility
           return U
