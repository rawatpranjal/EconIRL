# Context

MCE-IRL recovers a reward function by matching the occupancy measure of a
maximum-causal-entropy policy to the occupancy measure observed in
demonstrations. The linear variant parameterizes the reward as a dot product
with fixed features - fast, interpretable, but limited to the span of those
features. Deep MCE-IRL replaces that dot product with a small feedforward
network, allowing the estimator to fit reward surfaces that no fixed linear
basis would represent well.

## Source Ideas

The maximum causal entropy framework is from {ref}`Ziebart (2010) <ziebart-2010>`.
The key distinction from the earlier trajectory-based maximum entropy
formulation is causal: the agent acts before observing future randomness, so
the soft Bellman operator and the forward occupancy pass respect the
arrow of time in the decision problem. The planning side is therefore identical
to MCE-IRL regardless of how the reward is parameterized.

{ref}`Wulfmeier, Ondruska, and Posner (2015) <wulfmeier-2015>` introduced
the neural reward map into this framework. Their observation is that the MCE
gradient - empirical occupancy minus policy occupancy - flows through the reward
matrix entry by entry, so any differentiable reward parameterization can be
trained by passing that gradient back through the network. The econirl
implementation uses an MLP with a ReLU activation, an optional anchor action
that holds one action's reward column at zero, and an optional absorbing-state
row fixed at zero.

## Where Deep MCE-IRL Fits

Deep MCE-IRL sits in the behavioral IRL family alongside MCE-IRL, MaxEnt-IRL,
GLADIUS, AIRL, and IQ-Learn. Its defining properties are:

- **Planning side**: identical to MCE-IRL - soft Bellman, known transitions,
  discounted occupancy matching.
- **Reward side**: a neural map rather than a linear table, so the reward is
  not identified to a unique parameter vector. The validated object is the
  anchored reward matrix and the behavior it implies.
- **Gauge**: the reward is anchored either by fixing one action's column to
  zero or by fixing an absorbing state's row to zero. Without an anchor the
  reward map is identified only up to action-independent shifts.
- **No structural parameters**: raw network weights are not a structural
  estimand. A finite parameter projection is available but should be interpreted
  with care (see the [Pre-Estimation Checks](pre_estimation.md) page for
  projection identification conditions).

Against MCE-IRL the trade is capacity versus interpretability. Against GLADIUS
the difference is planning method: Deep MCE-IRL uses an explicit soft Bellman
solve per epoch, while GLADIUS trains Q and EV networks with a Bellman
consistency penalty and does not require transitions to be supplied.
