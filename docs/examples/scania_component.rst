SCANIA Component X Replacement
===============================

.. image:: /_static/scania_component_overview.png
   :alt: SCANIA Component X replacement model showing degradation distribution, replacement probability curve, and sample vehicle trajectories with replacement resets.
   :width: 100%

This example applies structural estimation to a heavy truck component replacement problem inspired by the SCANIA IDA 2024 Industrial Challenge dataset. A fleet manager observes the degradation level of a truck component each period and decides whether to keep operating or replace it. Replacing incurs a fixed cost but resets degradation to zero. Continued operation incurs a cost that grows with degradation. The decision problem is structurally identical to Rust (1987) with degradation replacing mileage as the state variable.

The state space has 50 degradation bins and 2 actions (keep or replace). The default parameters produce a per-period replacement rate of roughly 2 percent, consistent with the SCANIA data where 2,272 out of 23,550 vehicles received a repair during the study period. When the real SCANIA CSV files are available the loader constructs a degradation index from the 14 anonymized operational readout variables. Otherwise it generates synthetic data with matching structure.

Quick start
-----------

.. code-block:: python

   from econirl.environments.scania import ScaniaComponentEnvironment
   from econirl.datasets import load_scania

   env = ScaniaComponentEnvironment()
   panel = load_scania(as_panel=True)

   # Or with real SCANIA data from Kaggle
   panel = load_scania(data_dir="path/to/scania_csvs/", as_panel=True)

Estimation
----------

NFXP and CCP estimators recover the structural parameters from the synthetic panel. The replacement cost is well identified because replacement events provide a clear signal. The operating cost gradient is harder to pin down with low replacement rates, which is a known feature of Rust-style models.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_scania

   df = load_scania()

   nfxp = NFXP(discount=0.9999).fit(
       df, state="degradation_bin", action="replaced", id="vehicle_id"
   )
   ccp = CCP(discount=0.9999, num_policy_iterations=20).fit(
       df, state="degradation_bin", action="replaced", id="vehicle_id"
   )

Low-level estimation pipeline
------------------------------

The environment and estimator classes give full control over transitions, features, and optimizer settings.

.. code-block:: python

   from econirl.environments.scania import ScaniaComponentEnvironment
   from econirl.datasets import load_scania
   from econirl.estimation.nfxp import NFXPEstimator, estimate_transitions_from_panel
   from econirl.preferences.linear import LinearUtility

   env = ScaniaComponentEnvironment(discount_factor=0.9999)
   panel = load_scania(as_panel=True)
   utility = LinearUtility.from_environment(env)
   transitions = estimate_transitions_from_panel(panel, num_states=50, max_increment=2)

   nfxp = NFXPEstimator(optimizer="BHHH", inner_solver="policy", compute_hessian=True)
   result = nfxp.estimate(panel, utility, env.problem_spec, transitions)
   print(result.summary())

Using real SCANIA data
----------------------

The real SCANIA Component X dataset is available through the IDA 2024 Industrial Challenge on Kaggle. Download the training files and pass the directory path to the loader. The loader normalizes the 14 operational readout variables, computes a mean degradation index, and constructs the replacement action from the time-to-event file.

.. code-block:: bash

   # Download from Kaggle, then:
   python examples/scania-component/scania_nfxp.py --data-dir /path/to/scania/

Running the example
-------------------

.. code-block:: bash

   # Synthetic data (no download needed)
   python examples/scania-component/scania_nfxp.py

   # Limit to 100 vehicles for quick testing
   python examples/scania-component/scania_nfxp.py --max-vehicles 100

   # With real SCANIA data
   python examples/scania-component/scania_nfxp.py --data-dir /path/to/scania/

Reference
---------

SCANIA Component X dataset, IDA 2024 Industrial Challenge.

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.
