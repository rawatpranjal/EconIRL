==============
API References
==============

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

Inverse reinforcement learning estimators
----------------------------------------------------------

Reward-learning estimators. ``GLADIUS`` and ``AIRL`` are the neural-reward
estimators; they are also importable as ``NeuralGLADIUS`` and ``NeuralAIRL``.

.. autosummary::
   :toctree: generated/

   MCEIRL
   MCEIRLNeural
   AIRL
   NeuralAIRL
   AIRLHet
   GLADIUS
   NeuralGLADIUS

Contrib estimators
----------------------------------------------------------

Research and baseline estimators outside the core roster, importable from
``econirl.contrib`` (for example ``from econirl.contrib import TDCCP``).

.. currentmodule:: econirl.contrib

.. autosummary::
   :toctree: generated/

   TDCCP
   MPEC
   NNES
   SEES
   UFXP
   RHIP
   IQLearn
   MaxEntIRL
   MaxMarginIRL
   NeuralUFXP

Environments and simulation
----------------------------------------------------------

Markov decision process environments and the panel simulator.

.. currentmodule:: econirl

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

Data preparation
----------------------------------------------------------

Panel validation and feature-identification checks.

.. autosummary::
   :toctree: generated/

   ~preprocessing.check_panel_structure
   ~preprocessing.feature_diagnostics

Datasets
----------------------------------------------------------

Bundled and downloadable dataset loaders.

.. autosummary::
   :toctree: generated/

   datasets
