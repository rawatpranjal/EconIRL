# Quick Start

## Important Links

- [AIRL Overview](../airl.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Applied Notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/airl/airl_applied_workflow.ipynb)
- [Counterfactuals](counterfactuals.md)

Use the public class for tabular state-only AIRL.

```python
from econirl import AIRL

model = AIRL(n_states=16, n_actions=4, compute_se=False)
print(model.n_states, model.n_actions, model.compute_se)
```

**Result**

```text
16 4 False
```

## Fit inputs

Call `model.fit(panel, transitions=transitions, reward=reward_spec)`.

The transition tensor must have shape `(n_actions, n_states, n_states)`. A
state-only `RewardSpec` has a feature array with shape
`(n_states, n_features)`. AIRL repeats those features over actions internally.

For a DataFrame, also pass `state=`, `action=`, `id=`, and `next_state=`. AIRL
does not infer a general transition system from the DataFrame.

## Uncertainty

Set `compute_se=True`, choose `n_bootstrap`, and set `se_seed`. AIRL resamples
whole individual trajectories. `conf_int()` returns percentile intervals for
centered state rewards and policy probabilities.

Raw reward weights do not receive structural coefficient intervals. The public
uncertainty target is the recovered reward surface and induced behavior.

## Next steps

After fitting, inspect `diagnostics_` and `summary()`. Use `predict_proba()` on
held-out states. Use `counterfactual(transitions=...)` to re-solve the policy
for the recovered reward under new dynamics. Pickle the fitted estimator only
after these checks pass.

The [applied notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/airl/airl_applied_workflow.ipynb)
runs this sequence against a built wheel.
