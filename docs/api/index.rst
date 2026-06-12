=============
API Reference
=============

This page lists the public ``econirl`` API. Each entry links to a page with the
full signature, constructor parameters, attributes, and methods. Import paths
match the listed name, for example ``from econirl import NFXP``.

Core types
----------------------------------------------------------

Problem specifications and panel data containers.

.. currentmodule:: econirl

.. autosummary::
   :toctree: generated/

   DDCProblem
   Panel
   Trajectory
   TrajectoryPanel
   RewardSpec
   SufficientStats

Structural estimators
----------------------------------------------------------

Linear-utility dynamic discrete choice estimators with statistical inference.

.. autosummary::
   :toctree: generated/

   NFXP
   CCP
   TDCCP
   UFXP
   NNES
   SEES

Inverse reinforcement learning estimators
----------------------------------------------------------

Reward-learning estimators. ``GLADIUS`` and ``AIRL`` are the neural-reward
estimators; they are also importable as ``NeuralGLADIUS`` and ``NeuralAIRL``.

.. autosummary::
   :toctree: generated/

   MaxEntIRL
   MaxMarginIRL
   MCEIRL
   MCEIRLNeural
   IQLearn
   GLADIUS
   AIRL

Environments and simulation
----------------------------------------------------------

Markov decision process environments and the panel simulator.

.. autosummary::
   :toctree: generated/

   RustBusEnvironment
   ~environments.ArrayMDP
   ~environments.random_mdp
   ~simulation.simulate_panel

Utilities and inference
----------------------------------------------------------

Reward parameterizations and the first-stage transition estimator.

.. autosummary::
   :toctree: generated/

   Utility
   LinearCost
   make_utility
   TransitionEstimator

Datasets
----------------------------------------------------------

Bundled and downloadable dataset loaders.

.. autosummary::
   :toctree: generated/

   datasets
