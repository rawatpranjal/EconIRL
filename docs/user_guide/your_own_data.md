# Loading Data and Pre-Estimation

EconIRL fits a panel of decisions. Each row is one observed choice: who decided,
the state they were in, and the action they took. This page takes a panel from
raw data to a fitted model and one counterfactual.

The examples use the bundled bus-engine dataset as a stand-in for your own panel.
Replace it with your DataFrame and the rest stays the same.

## What your data needs

One row per observed decision. Three columns name the pieces.

| Column | What it holds |
| --- | --- |
| `state` | The state when the decision was made, as an integer label. |
| `action` | The discrete action that was chosen, as an integer label. |
| `id` | The individual, unit, or trajectory the row belongs to. |

A `next_state` column is optional. When it is missing, the tabular estimators
read transitions from the order of states within each `id`.

## Check the assumptions first

Before you fit, your data has to fit the model. These estimators recover a
reward from choices under a short list of assumptions. Read them as a readiness
checklist, not a list of traps. If your problem satisfies them, the estimate
means what you want it to mean.

- **Markov transitions.** Where you land next depends only on the current state
  and the action you took, not on the whole past. If your real dynamics have
  memory, fold what matters into the state.
- **Additive choice shocks.** Each choice carries an extra private payoff the
  analyst does not see. These estimators model it as a logit shock added to the
  payoff. This is the standard discrete-choice setup.
- **Transitions estimated first.** The probability of each next state is supplied
  or estimated in a first step, separately from the reward. The reward fit treats
  it as given.
- **A reward anchor.** A reward is only recovered relative to a baseline. Fix one
  action to zero payoff, usually an exit or absorbing action, and hold the logit
  scale fixed. This pins the level and the scale so the rest of the reward is
  pinned down too.

The estimator pages spell these out per method. The [NFXP identification
section](../estimators/nfxp.md) is the worked example.

### Four questions that point to an estimator

The assumptions above let any of these estimators run. Four further questions narrow the
field to the one that fits your problem.

- **Is the reward state-only or action-dependent?** A state-only reward depends on where
  you are, not on what you do. It suits [AIRL](../estimators/airl.md) and
  [MCE-IRL](../estimators/mce_irl.md). When the payoff differs by action, a state-only
  reward cannot represent the action contrast, and the policy collapses toward uniform.
  Action-dependent rewards need a structural estimator ([NFXP](../estimators/nfxp.md),
  [CCP](../estimators/ccp.md)) or an anchored method
  ([AIRL-Het](../estimators/airl_het.md)). The
  [AIRL identification boundary](../estimators/airl/identification.md) works this through.
- **Are transitions deterministic or stochastic?** Structural estimators handle stochastic
  transitions directly. AIRL's reward-recovery guarantee holds under deterministic
  dynamics. Under stochastic dynamics AIRL still matches behavior, but the reward is no
  longer pinned. MCE-IRL's feature matching holds either way. The
  [AIRL page](../estimators/airl.md) states the condition.
- **Is the state low-dimensional or high-dimensional?** A handful of known features
  supports a linear reward: NFXP, CCP, MCE-IRL. A high-dimensional state, or a reward
  whose form you do not know, calls for a neural reward:
  [Neural MCE-IRL](../estimators/deep_mce_irl.md) or
  [GLADIUS](../estimators/gladius.md). Structural estimators stay linear by design.
- **Can you anchor a reward value?** A reward is recovered relative to a baseline. If you
  can pin one value, such as a known zero-payoff exit action R(s, exit) = 0, you fix the
  level. An anchor also lets you identify an action-dependent reward where a plain method
  cannot. AIRL-Het uses two anchors, one on the exit action and one on the absorbing
  state.

## Run the diagnostics

Three checks tell you whether the data can support a fit. Each one says what it
means and what to do if it fails.

### Is the panel well formed?

```python
from econirl.preprocessing import check_panel_structure

report = check_panel_structure(df, id_col="bus_id", period_col="period",
                               state_col="mileage_bin", action_col="replaced")
print(report.valid, report.n_individuals, report.n_observations)
print(report.warnings)
```

```text
True 90 9410
['Unbalanced panel: 80-120 periods per individual']
```

`report.valid` is `True`, so the panel is usable. The warning notes that
individuals are observed for different lengths, which these estimators allow. If
`report.valid` is `False`, read `report.warnings` and fix the flagged columns
before going on.

### Does every state show more than one action?

Look for states where only one action ever appears.

```python
counts = df.groupby("mileage_bin")["replaced"].nunique()
print((counts < 2).sum(), "of", df["mileage_bin"].nunique(), "states show one action")
```

```text
14 of 52 states show one action
```

What it means. A state where only one action is ever taken carries no choice
contrast at that state.

What to do. CCP and UFXP read action frequencies directly, so they need more than
one action at each state to fit. NFXP pools every observation through the
likelihood, so single-action states do not break it. They just add no
information. If many states are single-action, prefer NFXP, or collect more
varied data.

### Can the reward be recovered from choices?

A feature that never changes with the action cancels out of every choice
comparison, so its parameter cannot be recovered. The check is the action
contrast, not the raw design. `feature_diagnostics` reports both.

```python
import numpy as np
from econirl.preprocessing import feature_diagnostics

S, A = 20, 3
rng = np.random.default_rng(0)
phi = np.zeros((S, A, 2))
phi[:, :, 0] = rng.normal(size=(S, A))   # varies with the action
phi[:, :, 1] = rng.normal(size=(S, 1))   # same for every action in a state

diag = feature_diagnostics(phi)
print({k: round(v, 2) if isinstance(v, float) else v for k, v in diag.items()})
```

```text
{'num_features': 2, 'feature_rank': 2, 'condition_number': 1.29,
 'contrast_rank': 1, 'contrast_condition_number': 1.0}
```

Read three things from this.

- **`contrast_rank` against `num_features`.** Here the raw design is full rank,
  but `contrast_rank` is 1, not 2. The second feature is constant across actions,
  so one parameter cannot be recovered. When `contrast_rank` is below
  `num_features`, some reward parameters are not identified. Drop the
  unidentified feature, or change the reward form so every feature moves with the
  action.
- **`contrast_condition_number`.** Even at full rank, a high condition number
  means the fit is numerically fragile. Features whose effects span orders of
  magnitude are close to unidentified on a small panel: the data cannot tell them
  apart. Rescale the features to comparable magnitudes, or simplify the reward.
- **`condition_number`.** The same reading for the raw design. A large value
  warns of instability before optimization even starts.

Only fit once `contrast_rank` equals `num_features` and both condition numbers
are moderate.

(choosing-your-estimator)=
## Choosing your estimator

The four questions above point to a family. Two families fit the same kind of panel but
recover different objects. Pick by what you need from the answer.

The key decisions, in short:

- **Need parameters with units and counterfactuals?** Use a structural estimator.
- **Reward state-only and dynamics deterministic?** AIRL or MCE-IRL recover it.
- **State high-dimensional or reward form unknown?** Use a neural reward.
- **Have a credible anchor action?** It pins the level and widens what you can identify.

| Family | Estimators | Recovers | Counterfactuals | Assumptions |
| --- | --- | --- | --- | --- |
| Structural | NFXP, CCP, MPEC | Finite reward parameters with a fixed meaning. | Supported: re-solve the model under a changed primitive. | Stronger: a parametric reward and the anchor above. |
| IRL / behavioral | MCE-IRL, AIRL | A reward only up to behavior-preserving transformations. | Not always: some methods recover behavior without a re-solvable primitive. | Weaker: less structure on the reward. |

Structural estimators are the choice when you need parameters with units and want to ask
what happens under a new policy. IRL estimators trade that for weaker assumptions. They
recover a reward that matches behavior but is identified only up to transformations that
leave behavior unchanged.

When the reward is state-only and the dynamics are deterministic, MCE-IRL and AIRL recover
the reward up to an additive constant, with linear features or a neural reward. An anchor
that pins one reward value fixes the constant too, so the recovered reward equals the true
reward. See the [estimator map](../estimators.md) for the full set and
[Choosing an Estimator](../estimators/landscape.md) for the side-by-side comparison.

## A full run, start to finish

This is the whole path on the bundled panel: load, diagnose, choose, fit, read
the outputs, run one counterfactual.

```python
import numpy as np
from econirl import NFXP
from econirl.datasets import load_rust_bus
from econirl.preprocessing import check_panel_structure

# 1. Load the panel. Swap in your own DataFrame here.
df = load_rust_bus()

# 2. Diagnose.
report = check_panel_structure(df, id_col="bus_id", period_col="period",
                               state_col="mileage_bin", action_col="replaced")
print(report.valid)

# 3. Choose. We want reward parameters and a counterfactual, so structural.
model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")

# 4. Fit.
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
print(model.params_)

# 5. Counterfactual: raise the replacement cost, read the new policy.
cf = model.counterfactual(RC=4.0)
print("replace probability at state 50:", float(np.asarray(cf.policy)[50, 1]))
```

```text
True
{'theta_c': 0.0010029257006533541, 'RC': 3.072263842893654}
replace probability at state 50: 0.055196291500871957
```

## Read the outputs

A fitted estimator exposes a common set of attributes.

| Attribute | What it holds |
| --- | --- |
| `params_` | Estimated reward or cost parameters. |
| `policy_` | Estimated choice probabilities, one row per state. |
| `value_` | Estimated value function, when the estimator computes one. |
| `summary()` | A report of the estimates with standard errors. |

## One counterfactual

Change a parameter and read the new policy.

```python
cf = model.counterfactual(RC=4.0)
print(cf.params)
print("replace probability at state 50:", float(np.asarray(cf.policy)[50, 1]))
```

```text
{'theta_c': 0.0010029257006533541, 'RC': 4.0}
replace probability at state 50: 0.055196291500871957
```

A higher replacement cost lowers the chance of replacing at a given mileage.

Not every estimator supports a counterfactual call. The structural family
re-solves the model under a changed primitive. Several IRL methods recover
behavior without a re-solvable primitive, so they have no `.counterfactual()`.
See [Choosing your estimator](#choosing-your-estimator) above.

## Next steps

- Pick an estimator for your problem in the [estimator map](../estimators.md).
- Read the [overview](overview.md) for the ideas behind these models.
- Look up any class in the [API reference](../api/index.rst).
