econirl
=======

EconIRL is a Python package for structural dynamic discrete choice and inverse
reinforcement learning. It helps teams estimate forward-looking choice models,
recover interpretable rewards, and evaluate counterfactual policies from panel
data.

Start by choosing an estimator. Then fit the NFXP reference model or use the
estimator map to match a method to your data and validation needs.

Quick Installation
------------------

.. code-block:: bash

   pip install econirl

Choose an Estimator
-------------------

.. raw:: html

   <details open>
     <summary><strong>Structural Econometrics</strong></summary>
     <ul>
       <li><a href="estimators/nfxp.html">NFXP</a>. Exact tabular dynamic discrete choice.</li>
       <li><a href="estimators/ccp.html">CCP and NPL</a>. Hotz-Miller inversion with policy iteration.</li>
       <li><a href="estimators/mpec.html">MPEC</a>. Constrained likelihood estimation.</li>
       <li><a href="estimators/sees.html">SEES</a>. Sieve value-function structural estimation.</li>
       <li><a href="estimators/nnes.html">NNES</a>. Neural value approximation inside NPL.</li>
       <li><a href="estimators/tdccp.html">TD-CCP</a>. Transition-free CCP estimation.</li>
     </ul>
   </details>
   <details>
     <summary><strong>Inverse Reinforcement Learning</strong></summary>
     <ul>
       <li><a href="estimators/mce_irl.html">MCE-IRL</a>. Maximum causal entropy reward-feature matching.</li>
       <li><a href="estimators/deep_mce_irl.html">Deep MCE-IRL</a>. Neural reward-map recovery.</li>
       <li><a href="estimators/airl.html">AIRL</a>. Adversarial state-reward recovery.</li>
       <li><a href="estimators/airl_het.html">AIRL-Het</a>. Latent-segment adversarial recovery.</li>
       <li><a href="estimators/f_irl.html">f-IRL</a>. f-divergence state-marginal matching.</li>
       <li><a href="estimators/gladius.html">GLADIUS</a>. Projected reward analysis from Q models.</li>
       <li><a href="estimators/iq_learn.html">IQ-Learn</a>. Inverse soft-Q learning diagnostics.</li>
     </ul>
   </details>

Quick Example
-------------

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

Estimator Guides
----------------

.. toctree::
   :maxdepth: 2

   estimators

First Steps
-----------

.. toctree::
   :maxdepth: 2

   user_guide/overview
   user_guide/api_design
   user_guide/validation
