econirl
=======

EconIRL is a Python package for structural dynamic discrete choice and inverse
reinforcement learning. Use it to estimate forward-looking choice models,
recover rewards, and predict behaviour under policy changes.

Install
-------

You can install EconIRL via:

.. code-block:: bash

   pip install econirl

Estimators
----------

Start with `Choosing an Estimator <estimators/landscape.html>`__ for the
canonical NFXP case, reward targets, transitions, and identification strategies.
Then open a method page below for its assumptions, evidence, and examples.

`NFXP <estimators/nfxp.html>`__ ·
`CCP <estimators/ccp.html>`__ ·
`MPEC <estimators/mpec.html>`__ ·
`UFXP <estimators/ufxp.html>`__ ·
`NNES <estimators/nnes.html>`__ ·
`TD-CCP <estimators/tdccp.html>`__ ·
`MCE-IRL <estimators/mce_irl.html>`__ ·
`Neural MCE-IRL <estimators/deep_mce_irl.html>`__ ·
`AIRL <estimators/airl.html>`__ ·
`RHIP <estimators/rhip.html>`__ ·
`f-IRL <estimators/f_irl.html>`__ ·
`GLADIUS <estimators/gladius.html>`__ ·
`IQ-Learn <estimators/iq_learn.html>`__

Replications
------------

See `Replications <replications.html>`__ for the terse paper-number ledger.
That page is about direct paper-number comparisons; broader synthetic evidence
lives in the simulation studies.

Example
-------

Here is an example that estimates Rust (1987):

.. code-block:: python

   from econirl.datasets import load_rust_bus, rust_bus_reward_spec
   from econirl import NFXP

   df = load_rust_bus()
   reward = rust_bus_reward_spec(n_states=90)   # explicit features, your own names
   model = NFXP(n_states=90, discount=0.9999, utility=reward)
   model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

   print(model.summary())

The report shows data coverage, identification, estimates, uncertainty, and
model fit.

.. code-block:: text

   ================================================================================
                      Dynamic Discrete Choice Estimation Results
   ================================================================================
   Method:      NFXP (Nested Fixed Point)    Observations:  9,410
   Optimizer:   BHHH                         Individuals:   90
   Family:      structural (linear utility)  Discount (β):  0.9999
                                             Scale (σ):     1.0
                                             Date:          2026-07-25

   [1] DATA
     State space:          90 states x 2 actions
     Periods per individual: ~120
     Obs per state:        max 901 . p95 703 . p50 61 . p5 4 . min 2
     State coverage:       52/90 visited (58%)
     Single-action states: 14

   [2] PRE-ESTIMATION CHECKS
     Reward features (K):  2
     Design rank:          2/2         Condition:          5.2e+01
     Contrast rank:        2/2       Contrast condition: 1.0e+02
     Verdict:              identified -- every reward parameter varies across actions

   [3] FIRST-STAGE TRANSITION ESTIMATION
     Transition source: estimated from fitted panel
     Method:               empirical frequencies (multinomial MLE)
     Transitions used:     N = 9,410        Free parameters: 72
     Rows with full support: 90/180
     Std err across cells: max 0.2722 . p50 0.0258 . min 0.0000
     Held fixed in stage two (block-diagonal information).
   --------------------------------------------------------------------------------
   [4] RESULTS
     4a. Estimation
                            coef   std err       t   P>|t|   [0.025   0.975]
       operating_cost      0.0010    0.0004    2.58   0.010   0.0002   0.0018
       replacement_cost    3.0723    0.0740   41.54   0.000   2.9273   3.2172
     4b. Identification
       Hessian condition:  72,739.9     Min eigenvalue: 180.93
       Status:             Potentially weakly identified
     4c. Inference & fit
       Converged:   yes
       Iterations:  5
       Estimation time: 6.04 seconds
       Message:     Converged
       SE method:  robust (sandwich)
       Log-lik:    -1,900.33
       AIC/BIC:    3,804.7 / 3,819.0
       accuracy 94.9%
   ================================================================================

Counterfactuals
---------------

A fitted model evaluates policy changes. The counterfactual compares the
baseline and counterfactual policies over long-run demand and welfare. Raising
the replacement cost lowers the long-run replacement rate and lets buses run to
higher mileage.

.. code-block:: python

   cf = model.counterfactual(replacement_cost=4.0)   # raise the replacement cost
   print(cf.summary())

.. code-block:: text

   ==========================================================================
                             Counterfactual Summary
   ==========================================================================
   Type 3: reward parameter change      Oracle: none
   Change:  parameters [0.001, 3.072] -> [0.001, 4.0]
   --------------------------------------------------------------------------
                                 baseline  counterfactual    change
     Action rate a=0 (long-run)     0.947           0.971    +0.023
     Action rate a=1 (long-run)     0.053           0.029    -0.023
     Long-run state mean           10.502          16.612    +6.110
     Expected value  E_mu[V]      339.703          -2.482  -342.186
   --------------------------------------------------------------------------
     Policy shift |dpi|:   max 0.03 . p95 0.03 . p50 0.03 . p5 0.03
     Welfare change:       -342.29 utils (mean states) , -342.19 (stationary)
     (welfare = E[V], the inclusive value / consumer surplus)
   ==========================================================================

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/overview
   user_guide/quick_start
   user_guide/your_own_data
   estimators/core
   estimators/other
   comparing_estimators
   user_guide/post_estimation
   replications
   simulation_studies/index
   api/index
   references
