# Counterfactuals

## Important Links

- [MCE-IRL overview](../mce_irl.md)
- [Quick start](quick_start.md)
- [Pre-estimation checks](pre_estimation.md)
- [Simulation evidence](validation.md)

The fitted `MCEIRL` model can re-solve the dynamic program after one primitive
changes. Use a reward-parameter change or a transition change. Supplying both,
or neither, raises an error.

`model.counterfactual(params=...)` accepts a complete parameter array or a
dictionary containing the parameters to change. The result includes the
baseline and new policies, value functions, changes, and `welfare_change`, the
mean change in the fitted value function. Reward levels are not identified. Do
not interpret this field as identified welfare. Compare policy changes and
value differences after applying an explicit normalization.

`model.counterfactual(transitions=...)` accepts either a dense transition
tensor in `(A, S, S)` orientation or `DeterministicTransitions`. For a fit
without tasks, use the fitted state and action indexing. For a task-based fit,
pass global `DeterministicTransitions` in the original indexing so the
estimator can recompile the same task subgraphs. A dense task counterfactual
must already use the compiled task indexing.

## Counterfactual Families

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Change reward parameters and hold transitions fixed. | Payoff response. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics response. |
| Type C | Change the valid-action mask. | Choice-set response. |

The primary generated simulation evaluates all three families.

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.006456 | 0.000742 | 0.000433 |
| Type B | 0.006284 | 0.000523 | 0.000410 |
| Type C | 0.004211 | 0.000145 | 0.000094 |

These results apply to the simulated transition law and reward basis. A
counterfactual needs the same support and policy-response assumptions as the
fitted model.
