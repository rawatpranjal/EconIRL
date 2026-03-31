Quickstart
==========

This page shows how to fit a structural model in three steps. It assumes you know what dynamic discrete choice models and inverse reinforcement learning are.

Fitting a model
---------------

Load the Rust bus engine replacement dataset and fit two estimators.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp  = CCP(discount=0.9999, num_policy_iterations=5).fit(df, state="mileage_bin", action="replaced", id="bus_id")

Both estimators recover the same parameters. NFXP uses nested fixed-point maximum likelihood. CCP uses Hotz-Miller inversion with nested pseudo-likelihood iterations. CCP is faster on this problem because it avoids the inner Bellman loop.

.. code-block:: python

   print(nfxp.summary())
   print(ccp.summary())
   print(nfxp.params_)   # {'theta_c': 0.001, 'RC': 3.07}
   print(nfxp.se_)        # {'theta_c': 0.00003, 'RC': 0.12}
   print(nfxp.conf_int()) # {'theta_c': (0.00094, 0.00106), 'RC': (2.83, 3.31)}

Predicting choices
------------------

Every fitted estimator can predict choice probabilities for new states.

.. code-block:: python

   import numpy as np
   proba = nfxp.predict_proba(np.array([0, 30, 60, 89]))
   # proba[i, 0] = P(keep | mileage=i), proba[i, 1] = P(replace | mileage=i)

Custom features
---------------

To use your own data with custom features, create a RewardSpec.

.. code-block:: python

   import torch
   from econirl import NFXP, RewardSpec

   features = torch.randn(50, 3, 4)  # 50 states, 3 actions, 4 features
   spec = RewardSpec(features, names=["price", "quality", "distance", "brand"])

   model = NFXP(n_states=50, n_actions=3, discount=0.95)
   model.fit(panel, reward=spec, transitions=transitions)
   print(model.params_)
