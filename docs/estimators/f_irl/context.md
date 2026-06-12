# Context

Most IRL estimators frame learning as feature expectation matching: the reward
is parameterized by a small weight vector, and estimation finds the weight that
aligns the agent's expected features with the expert's. f-IRL takes a different
route. Instead of matching features, it matches distributions: the method
minimizes a divergence between the state marginal of the expert data and the
state marginal induced by the candidate reward.

## Source Ideas

The construction comes from {ref}`Ni et al. (2020) <ni-2020>`. The paper notes
that existing IRL methods are implicitly minimizing a divergence between
trajectory distributions under a fixed feature mapping, which ties the reward
estimate to the choice of features and may discard distributional information.
f-IRL makes the divergence objective explicit and leaves the divergence family
as a design choice (forward KL, reverse KL, Jensen-Shannon, chi-squared, or
total variation), allowing the practitioner to select the behavior that fits the
problem.

The paper validates on robotics tasks where the reward is stationary and the
agent's state marginal is the natural comparison object. The econirl
implementation carries this scope directly: the primary validated cell uses
state-marginal matching and a state-only reward.

## Where f-IRL Fits

f-IRL sits in the behavioral IRL family alongside MaxEnt-IRL, MCE-IRL, and
GLADIUS. It recovers a tabular reward function rather than structural utility
parameters, so recovered reward values are not in the same gauge as the
data-generating utility and standard errors are not reported.

Against MaxEnt-IRL and MCE-IRL the key difference is objective: those methods
match feature expectations, while f-IRL matches state marginals directly. This
means f-IRL does not require a feature specification, but it also means the
recovered reward is a non-parametric table rather than a low-dimensional
weight vector.

Against GLADIUS and AIRL the difference is architecture: f-IRL is tabular
and gradient-based, with no discriminator network and no adversarial training
loop. It is simpler to configure and inspect, at the cost of the scalability
those neural methods offer.

Action-dependent structural DDC reward recovery is outside the validated scope.
The action-dependent diagnostic cell fails the reward-range check; see the
[Simulation Study](validation.md) page for the full diagnostic record.
