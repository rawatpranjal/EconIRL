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
baseline and new policies, value functions, and their changes. Reward levels
are not identified. The `welfare_change` field is therefore `None`. The
returned `value_change` is a difference between model value arrays, not an
identified welfare estimate. Interpret it only under an externally justified
reward normalization. Use policy changes when no such normalization is
available.

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
| Type C | Supply `DeterministicTransitions` with a changed valid-action mask and hold rewards fixed. | Choice-set response. |

The [simulation study](validation.md) evaluates all three families.

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.006456 | 0.000742 | 0.000433 |
| Type B | 0.006284 | 0.000523 | 0.000410 |
| Type C | 0.004211 | 0.000145 | 0.000094 |

In this simulation, Value RMSE measures recovery against known model values
under a fixed reward normalization. It is not an identified welfare estimate.
The results apply to the simulated transition law and reward basis.
Counterfactual interpretation also requires the support and policy-response
assumptions of the fitted model.
