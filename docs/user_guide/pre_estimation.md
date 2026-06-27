# Pre-Estimation Diagnostics

Run these checks before treating any estimate as structural evidence. A clean fit
on a rank-deficient or poorly conditioned design is not identification.

## Check the panel

`check_panel_structure` validates the trajectory layout: one row per
agent-period, valid successor states, and adequate action support.

```python
from econirl.preprocessing import check_panel_structure

report = check_panel_structure(df, id_col="bus_id", state_col="mileage_bin", action_col="replaced")
print(report)
```

## Check the reward design

`feature_diagnostics` reports the rank and conditioning of the reward-feature
matrix, both raw and action-differenced. State-only features that do not vary
across actions difference out of the choice probabilities and leave the
parameters on a ridge, so the action-differenced rank is the one that matters.

```python
from econirl.preprocessing import feature_diagnostics

diag = feature_diagnostics(feature_matrix)
print(diag)
```

## What to confirm

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of parameters means the reward is not identified. |
| Feature condition number | A high condition number signals unstable estimates. |
| Action-contrast rank | State-only features difference out, so check the action-differenced rank, not the raw rank. |
| Transition row sums | Each transition row must be a valid probability distribution. |
| Transition orientation | Estimator-facing code expects the (actions, states, states) orientation. |
| State coverage | Unobserved states produce weak or degenerate likelihood regions. |
| Action support | Rare or single-action states leave their payoff weakly identified. |
| Reward normalization | The reward level and scale need a valid anchor. |

Stop if the action-contrast rank is below the number of parameters. Each
estimator's page carries the same checklist specialized to its assumptions.
