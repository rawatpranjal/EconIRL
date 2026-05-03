Estimators
==========

All estimators follow the same interface. Call ``fit()`` with a DataFrame, then access ``params_``, ``se_``, ``policy_``, and ``summary()``.

Linear Reward Estimators
------------------------

.. autoclass:: econirl.estimators.nfxp.NFXP
   :members: fit, summary, predict_proba, conf_int, simulate, counterfactual
   :undoc-members:

.. autoclass:: econirl.estimators.nnes.NNES
   :members: fit, summary, predict_proba, conf_int
   :undoc-members:

.. autoclass:: econirl.estimators.ccp.CCP
   :members: fit, summary, predict_proba, conf_int
   :undoc-members:

.. autoclass:: econirl.estimators.tdccp.TDCCP
   :members: fit, summary, predict_proba, conf_int
   :undoc-members:

Neural Reward Estimators
------------------------

.. autoclass:: econirl.estimators.neural_gladius.NeuralGLADIUS
   :members: fit, summary, predict_proba, predict_reward, conf_int
   :undoc-members:

.. autoclass:: econirl.estimators.neural_airl.NeuralAIRL
   :members: fit, summary, predict_proba, predict_reward, conf_int
   :undoc-members:

Protocol
--------

.. autoclass:: econirl.estimators.protocol.EstimatorProtocol
   :members:
