# Counterfactuals

## Important Links

- [Neural MCE-IRL overview](../deep_mce_irl.md)
- [Quick Start](quick_start.md)
- [Simulation Study](validation.md)
- [Post-Estimation Guide](../../user_guide/post_estimation.md)

`counterfactual()` re-solves the soft Bellman problem after one primitive
changes. It does not retrain the neural network.

| Input | Change |
| --- | --- |
| `reward_delta=` | Add a reward perturbation, then restore the fitted normalization |
| `transitions=` | Replace the transition tensor |
| `action_mask=` | Remove actions that were available during fitting |

Supply exactly one input. The result contains baseline and changed policies,
values, their differences, and the average value change.

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
must retain the same tensor shape. Every state must keep at least one available
action.

The reward map remains identified only under its fitted normalization. Compare
counterfactual values within a fitted model, not across separate fits.
