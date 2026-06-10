# Counterfactuals

The public `MCEIRL` wrapper exposes the recovered policy, reward matrix, and
value function. It does not yet provide a one-call `counterfactual` method like
the structural likelihood wrappers.

The counterfactual evidence comes from the known-truth validation harness. The
harness reruns the dynamic program under controlled changes and compares the
recovered-reward policy with the oracle policy. See the
[validation page](validation.md) for the generator script, table source, and
JSON artifact.

## Counterfactual Families

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift rewards and hold transitions fixed. | Payoff counterfactual. |
| Type B | Change transitions and hold rewards fixed. | State-dynamics counterfactual. |
| Type C | Disable one non-anchor action. | Action-set or design counterfactual. |

## Reported Results

These rows are from the primary `mce_low_high_reward` known-truth artifact.

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.006456 | 0.000157 | 0.000742 | 0.000433 |
| Type B | 0.006284 | 0.000142 | 0.000523 | 0.000410 |
| Type C | 0.004211 | 5.98e-5 | 0.000145 | 0.000094 |

Regret reports how the policy induced by the recovered reward compares with the
oracle counterfactual policy.

## API Boundary

For package users, the stable public objects are the fitted reward, policy, and
value arrays. For controlled payoff, transition, or action-set interventions,
use the lower-level simulation and evaluation utilities with an explicit
problem and transition environment.
