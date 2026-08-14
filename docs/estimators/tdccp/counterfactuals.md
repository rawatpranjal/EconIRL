# Counterfactuals

## Important Links

- [TD-CCP overview](../tdccp.md)
- [Quick Start](quick_start.md)
- [Evidence](validation.md)
- [Applied workflow notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/tdccp/tdccp_applied_workflow.ipynb)

TD-CCP avoids a transition density in the reward-parameter stage.
Counterfactual evaluation is a separate task. It needs a transition environment
because the changed dynamic program must be solved.

## Reward Changes

Pass one or more fitted parameter names to `counterfactual`. The method holds
the stored transition tensor fixed, changes the reward, and solves for a new
policy and value function.

The following continues the model from [Quick Start](quick_start.md).

```python
changed = model.counterfactual(
    replacement_cost=model.params_["replacement_cost"] - 0.25
)

print(f"{model.policy_[:, 1].mean():.3f}")
print(f"{changed.policy[:, 1].mean():.3f}")
print(f"{np.abs(changed.policy - model.policy_).mean():.3f}")
```

**Result**

```text
0.463
0.493
0.030
```

The lower fitted replacement cost raises the mean replacement probability.
The last line is the mean absolute change over all state-action probabilities.

## Transition Changes

Pass a new tensor through `transitions=` to hold fitted reward parameters fixed
and change the state dynamics. The tensor must have shape
`(n_actions, n_states, n_states)`. Every row must be finite, nonnegative, and
sum to one.

Reward and transition changes are separate calls. Mixing both in one call is
rejected so the reported intervention remains clear.

## Returned Quantities

The result provides both baseline and changed objects.

| Attribute | Meaning |
| --- | --- |
| `baseline_policy` | Policy under the fitted model |
| `counterfactual_policy` | Policy after the intervention |
| `baseline_value` | Value under the fitted model |
| `counterfactual_value` | Value after the intervention |
| `params` | Parameter values used for the changed model |

The shared aliases `policy` and `value_function` point to the changed policy
and value.

## Interpretation Boundary

The simulation evidence first measures the policy effect under the oracle
intervention. It then compares the fitted counterfactual with the oracle
changed policy and reports value regret. This separates the size of the
intervention from counterfactual recovery error.

TD-CCP does not infer a new transition law from a reward change. A transition
intervention must be supplied explicitly or constructed from a separate model.
