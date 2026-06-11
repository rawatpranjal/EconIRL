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

.. raw:: html

   <details open>
     <summary><strong>Structural Econometrics</strong></summary>
     <ul>
       <li><a href="estimators/nfxp.html">NFXP</a></li>
       <li><a href="estimators/ccp.html">CCP and NPL</a></li>
       <li><a href="estimators/mpec.html">MPEC</a></li>
       <li><a href="estimators/nnes.html">NNES</a></li>
       <li><a href="estimators/tdccp.html">TD-CCP</a></li>
     </ul>
   </details>
   <details>
     <summary><strong>Inverse Reinforcement Learning</strong></summary>
     <ul>
       <li><a href="estimators/mce_irl.html">MCE-IRL</a></li>
       <li><a href="estimators/deep_mce_irl.html">Deep MCE-IRL</a></li>
       <li><a href="estimators/airl.html">AIRL</a></li>
       <li><a href="estimators/airl_het.html">AIRL-Het</a></li>
       <li><a href="estimators/f_irl.html">f-IRL</a></li>
       <li><a href="estimators/gladius.html">GLADIUS</a></li>
       <li><a href="estimators/iq_learn.html">IQ-Learn</a></li>
     </ul>
   </details>

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

   {'theta_c': 0.0010028828858836278, 'RC': 3.0722093435989524}
   0.05519477716656161

.. toctree::
   :hidden:
   :maxdepth: 2

   estimators

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/overview
   user_guide/api_design
   user_guide/validation
