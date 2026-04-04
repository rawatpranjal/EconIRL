Troubleshooting
===============

This page covers common issues and how to resolve them.

NFXP Not Converging
--------------------

The NFXP estimator uses nested fixed point iteration to solve the Bellman equation inside a likelihood optimizer. If the inner loop does not converge, the outer optimizer receives inaccurate gradients and may fail to find the maximum.

The most common cause is unnormalized features. When state features span different scales, the reward function parameters become poorly conditioned and the Bellman operator contracts slowly. Normalize all features to the range negative one to one before estimation.

A high discount factor also slows convergence. With discount factors above 0.99, the contraction rate of the Bellman operator approaches one and requires many more iterations. For initial testing and debugging, try a lower discount factor like 0.95 or 0.99. Once the pipeline works, increase to the target value and raise ``inner_max_iter`` accordingly.

If the optimizer oscillates without progress, check that transitions are properly normalized. Every row of the transition matrix must sum to one. The ``TransitionEstimator`` handles this automatically when fitting from data, but manually constructed transition matrices sometimes have rounding errors.

MCE-IRL Oscillating
--------------------

Maximum Causal Entropy IRL can oscillate when the learning rate is too high or when features have very different scales. The gradient update overshoots the optimum and bounces back, producing a loss curve that does not settle.

Reduce the learning rate by a factor of ten and check whether the loss curve stabilizes. If features have different magnitudes, apply the ``RunningNorm`` preprocessor from ``econirl.preprocessing`` to standardize them during training. The default learning rate of 0.01 works well for normalized features but may be too aggressive for raw data.

The inner loop of MCE-IRL runs soft value iteration to compute the current policy. If the discount factor is high and ``inner_max_iter`` is too low, the inner loop may not converge, producing an inaccurate policy gradient. Increase ``inner_max_iter`` until the inner loop residual drops below the tolerance.

Standard Errors Invalid
------------------------

Standard errors require the Hessian of the log-likelihood to be negative definite at the optimum. If the Hessian is singular or nearly singular, the estimated variance-covariance matrix will have negative diagonal entries or extremely large values, producing meaningless confidence intervals.

Check the condition number of the Hessian using ``result.identification.condition_number`` from the ``EstimationSummary``. Condition numbers above one million indicate near-singularity. This usually means one or more parameters are not identified. Common causes include collinear features, a state space that is too small to identify all parameters, or a discount factor that is too close to one.

If identification diagnostics flag a problem, consider reducing the number of parameters, adding more data, or imposing parameter restrictions. The ``IdentificationDiagnostics`` object reports eigenvalues of the Hessian. Near-zero eigenvalues indicate the directions in parameter space that are poorly identified.

JAX Float64 Warnings
---------------------

Structural estimation requires 64-bit floating point precision for inner loop tolerances of 1e-12 and accurate Hessian computation. JAX defaults to 32-bit precision, which silently truncates these computations and produces inaccurate results.

econirl automatically enables 64-bit precision when you import the package. If you see warnings about float64 truncation, it usually means JAX was imported and used before econirl was imported. Move ``import econirl`` to the top of your script, before any other JAX operations.

You can verify that 64-bit precision is active by running ``import jax; assert jax.config.jax_enable_x64``. If this assertion fails after importing econirl, file a bug report.

GLADIUS Operating Cost Bias
-----------------------------

When GLADIUS is used in the IRL setting without observed rewards, it recovers replacement cost within 8 percent but systematically overestimates operating cost by about 40 percent. This is not a tuning problem. It is a structural identification limitation.

GLADIUS trains Q-values by negative log-likelihood only, which identifies Q up to a state-dependent constant. That constant propagates differently through the transition structure for different actions. On the bus engine problem, maintaining a bus stays near the current state while replacing jumps to state zero. The asymmetric propagation produces a systematic bias in the implied operating cost.

This bias persists regardless of network size, training duration, or data volume. If you need accurate operating cost recovery, use NFXP or CCP instead. GLADIUS is best suited for continuous-state environments where tabular structural methods cannot be applied, or when rewards are observed in the data and can anchor Q-values through the bi-conjugate Bellman error.

IRL Reward Scale Ambiguity
---------------------------

Inverse reinforcement learning recovers rewards only up to additive constants and multiplicative scale. Two reward functions that differ by a constant or a positive scalar produce identical optimal policies and identical choice probabilities. This is a fundamental identification result, not a limitation of any particular estimator.

When comparing IRL-recovered rewards to true parameters, do not use raw RMSE. Instead use cosine similarity between parameter vectors, which is invariant to scale, or compare policy-level outcomes like choice probabilities and welfare. The ``EstimationSummary`` reports cosine similarity when true parameters are provided.

If you need rewards in levels rather than up to scale, you must impose additional restrictions. Common approaches include normalizing one parameter to a known value, using observed rewards to anchor the scale, or imposing a known outside option value. Without such restrictions, only reward differences across actions at a given state are identified.
