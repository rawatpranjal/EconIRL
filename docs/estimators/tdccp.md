# TD-CCP

TD-CCP estimates reward parameters in dynamic discrete choice models from
panel data. It uses observed choices, states, and next states to learn the
continuation terms that usually require a transition-density model.

The estimator is useful when the reward is a finite set of known features, but
the state process is awkward to model directly. The parameter-estimation step
does not fit a transition-density model. If you later want policy values or
counterfactuals, you still need a transition environment for that evaluation
step.

TD-CCP is not a general neural reward-recovery method. In EconIRL, the
reported study is narrower and more precise: a finite reward parameter vector
is recovered in a synthetic data simulation with the locally robust, cross-fitted
estimator described in the TD-CCP paper.

## When To Use It

Use TD-CCP when choices are discrete, agents are forward-looking, and you have
panel trajectories with current and next state-action information. It is a good
fit when transition-density modeling is the difficult part of the problem, but
the reward can be written as a finite linear function of known features.

Prefer another estimator when the state space is small and tabular likelihood
methods are easy to run, when observed action support is sparse, or when the
target is an unrestricted neural reward map rather than structural reward
parameters.

## Minimal Fit

```python
from econirl.datasets import load_rust_bus
from econirl import TDCCP

df = load_rust_bus()

model = TDCCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    method="semigradient",
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

This wrapper is enough for a quick Rust-style bus replacement example. For
custom reward features, panel objects, basis settings, cross-fitting, robust
standard errors, or supplied transition tensors, use
`econirl.estimation.TDCCPEstimator`.

## Simulation Study

The current simulation study is the
`shapeshifter_encoded_state_locally_robust` synthetic cell. It uses 81
states, 3 actions, two encoded state coordinates, and 6 reward parameters.
Action 0 is the baseline action, so its reward is normalized to zero.

The reported estimator uses:

| Component | Current setting |
| --- | --- |
| Recursive-term method | Semigradient TD with encoded features |
| First-stage choice model | Logit with degree-2 state features |
| Inference | Cross-fitting with locally robust standard errors |
| Covariance unit | Individual |
| Monte Carlo check | 25 repeated-seed replications |

The results file records the locally robust moment equation, the
correction recursion, the covariance, optimizer stationarity, and standard
error coverage. The raw neural reward case is retained as a diagnostic only;
it has no finite true reward parameter vector and is not part of the primary
finite-parameter study.

## Reading Guide

```{toctree}
:maxdepth: 2

tdccp/context
tdccp/quick_start
tdccp/under_the_hood
tdccp/pre_estimation
tdccp/validation
tdccp/counterfactuals
tdccp/rust_bus
```
