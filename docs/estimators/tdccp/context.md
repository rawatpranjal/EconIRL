# Context

Dynamic discrete choice models describe agents who choose among discrete
actions today while accounting for future consequences. Standard estimators
often need a model for how states move after each action. That transition
model can be the hardest part of the problem when states are continuous,
high-dimensional, or encoded through flexible features.

TD-CCP addresses the parameter-estimation part of that problem. It keeps the
conditional choice probability logic of Hotz-Miller style estimators, but it
learns the continuation terms directly from observed panel transitions. The
input is a sequence of current actions and states together with the next action
and next state.

## Structural Target

The paper's reward target is finite-dimensional. In words, each action-state
pair has known reward features, and the estimator learns the weights on those
features.

```text
u_theta(a, x) = z(a, x)' theta
```

This distinction matters. TD-CCP is designed to recover the finite parameter
vector `theta`; it is not evidence that arbitrary raw neural reward values can
be recovered from choices alone.

## Main Idea

A CCP likelihood needs terms that summarize future utility. Traditional
approaches often compute those terms from a transition model. TD-CCP instead
learns them from observed successor tuples.

There are two paper paths:

- Semigradient TD uses basis functions and solves projected TD equations.
- Approximate value iteration repeatedly solves prediction problems and can
  use flexible learners.

EconIRL currently certifies the semigradient path. The flexible AVI path is
available, but it is not the current release claim.

## Where It Fits

NFXP is the exact tabular reference because it solves the dynamic program
inside the likelihood. CCP is fast when choice probabilities and transition
objects are well supported. TD-CCP is the right tool when the reward target is
finite-dimensional but transition-density modeling is the bottleneck.

Counterfactual analysis remains a separate step. TD-CCP avoids estimating the
original transition density for reward-parameter estimation, but evaluating a
new policy or counterfactual environment still requires transition information.
