# Counterfactuals

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Wulfmeier-Shaped Study](wulfmeier_objectworld.md)
- [Post-Estimation Guide](../../user_guide/post_estimation.md)

`counterfactual()` re-solves the soft Bellman problem after one primitive
changes. It does not retrain the neural network.

| Input | Change |
| --- | --- |
| `reward_delta=` | Add a reward perturbation, then restore the fitted normalization |
| `transitions=` | Replace the transition tensor |
| `action_mask=` | Remove actions that were available during fitting |

Supply exactly one input. The result contains the baseline and changed policies
and values, together with their differences. It also contains `welfare_change`,
the unweighted mean of the state-level value changes.

```python
mask = np.ones((model.n_states, model.n_actions), dtype=bool)
mask[:, 1] = False
result = model.counterfactual(
    action_mask=mask,
    description="remove action 1",
)
print(np.round(result.policy[:, 1], 6).tolist())
print(result.metadata["changed_primitive"])
```

**Result**

```text
[0.0, 0.0, 0.0]
action_availability
```

Reward changes must use the fitted state and action space. Transition changes
must have shape `(n_actions, n_states, n_states)`. Their entries must be finite
and nonnegative. Each state-action row must sum to one. Every state must keep at
least one available action.

The reward map remains identified only under its fitted normalization. Compare
counterfactual values within a fitted model, not across separate fits.

When the model was fitted with `se_method="bootstrap"`,
`result.metadata["bootstrap_intervals"]` contains percentile intervals. The
intervals cover the mean policy total variation and the mean value change under
the requested intervention. Each interval is computed by applying the same
intervention and re-solving the model for every successful fitted reward draw.
