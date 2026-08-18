# Counterfactuals

GLADIUS re-solves a soft Bellman model after an intervention. This is a
structural operation, so the estimator requires a valid known-reward anchor and
a row-stochastic transition tensor stored at fit time. It refuses structural
counterfactuals for unanchored fits.

## Supported Changes

| Family | Changed primitive | Oracle-simulation primary-cell regret |
| --- | --- | ---: |
| Type A | Reward shift | 0.00291 |
| Type B | Transition kernel | 0.00814 |
| Type C | Action removal through a large payoff penalty | 0.00102 |

The primary qualification cell passes upstream raw reward, projected reward,
Q, value, and policy gates together with all three regret gates. These figures
therefore support the scoped interventions in that oracle-simulation design. They do
not make an unanchored reward level identified or validate extrapolation beyond
the observed state and action support.

## Reward Intervention

```python
reward_delta = np.zeros_like(model.reward_)
reward_delta[:, 1] = 0.25
result = model.counterfactual(reward_delta=reward_delta)
```

`result.counterfactual_policy`, `result.counterfactual_value`,
`result.policy_change`, and `result.welfare_change` are computed from the
re-solved model. The supplied transition tensor is used here, not during
GLADIUS fitting.

Inspect `model.diagnostics_` before interpreting a result. Thin state-action
coverage, a rank-deficient reward projection, or an invalid anchor remains an
identification problem even when optimization completed.
