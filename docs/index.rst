econirl
=======

EconIRL is a Python package for structural dynamic discrete choice and inverse
reinforcement learning. Use it to estimate forward-looking choice models,
recover rewards, and evaluate counterfactual policies.

Install
-------

.. code-block:: bash

   pip install econirl

Estimators
----------

Start with `Choosing an Estimator <estimators/landscape.html>`__, then open the
estimator that matches your decision problem.

`NFXP <estimators/nfxp.html>`__ ·
`CCP <estimators/ccp.html>`__ ·
`MPEC <estimators/mpec.html>`__ ·
`UFXP <estimators/ufxp.html>`__ ·
`NNES <estimators/nnes.html>`__ ·
`TD-CCP <estimators/tdccp.html>`__ ·
`MCE-IRL <estimators/mce_irl.html>`__ ·
`Neural MCE-IRL <estimators/deep_mce_irl.html>`__ ·
`AIRL <estimators/airl.html>`__ ·
`AIRL-Het <estimators/airl_het.html>`__ ·
`RHIP <estimators/rhip.html>`__ ·
`f-IRL <estimators/f_irl.html>`__ ·
`GLADIUS <estimators/gladius.html>`__ ·
`IQ-Learn <estimators/iq_learn.html>`__

Replications
------------

See `Replications <replications.html>`__ for the terse paper-number ledger.

Example
-------

.. code-block:: python

   from econirl.datasets import load_rust_bus
   from econirl import NFXP

   df = load_rust_bus()
   model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
   model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

   print(model.params_)
   cf = model.counterfactual(RC=4.0)
   print(cf.policy[50, 1])

Output
^^^^^^

.. code-block:: text

   {'theta_c': 0.0010029257006533541, 'RC': 3.072263842893654}
   0.055196291500871957

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/overview
   user_guide/quick_start
   user_guide/your_own_data
   estimators/core
   estimators/other
   user_guide/post_estimation
   replications
   simulation_studies/index
   api/index
   references
