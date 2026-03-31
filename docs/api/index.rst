API Reference
=============

This section provides detailed documentation for all public modules and classes
in econirl.

.. toctree::
   :maxdepth: 2

   core
   estimators
   environments
   preferences
   estimation
   inference
   simulation
   visualization
   datasets

Overview
--------

econirl is organized into the following modules:

**Core** (:mod:`econirl.core`)
   Fundamental types and algorithms: ``DDCProblem``, ``Panel``, ``Trajectory``,
   Bellman operators, and solvers.

**Environments** (:mod:`econirl.environments`)
   Dynamic discrete choice environments following the Gymnasium API.
   Includes ``RustBusEnvironment`` and base classes for custom environments.

**Preferences** (:mod:`econirl.preferences`)
   Utility function specifications. ``LinearUtility`` for linear-in-parameters
   utility, with support for custom specifications.

**Estimation** (:mod:`econirl.estimation`)
   Estimators for dynamic discrete choice models: ``NFXPEstimator`` (Nested Fixed Point),
   ``CCPEstimator`` (Conditional Choice Probability).

**Inference** (:mod:`econirl.inference`)
   Statistical inference: ``EstimationResult`` with standard errors, confidence
   intervals, hypothesis tests, and goodness-of-fit measures.

**Simulation** (:mod:`econirl.simulation`)
   Data generation and counterfactual analysis: ``simulate_panel``,
   ``counterfactual_policy``, ``compute_welfare_effect``.

**Visualization** (:mod:`econirl.visualization`)
   Plotting utilities for policies, value functions, and estimation results.

**Datasets** (:mod:`econirl.datasets`)
   Built-in datasets including ``load_rust_bus`` for the Rust (1987) data.
