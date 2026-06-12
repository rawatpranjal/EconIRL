# f-IRL

f-IRL learns a reward by minimizing an f-divergence between the expert's state
marginal and the state marginal induced by the current reward and transition
model. The primary validated scope is state-only reward recovery with
state-marginal matching. Action-dependent structural DDC recovery remains a
diagnostic exercise under current evidence.

Start here when state marginal matching is the study question, transitions are
available, and you want an f-divergence IRL baseline rather than a
likelihood-based structural estimator.

## Source Papers

This page draws on {ref}`Ni et al. (2020) <ni-2020>` for f-IRL and
state-marginal matching via f-divergence minimization.

## Quick Decision

| Use f-IRL when | Prefer another estimator when |
| --- | --- |
| The reward target is state-only. | You need action-dependent structural DDC reward recovery (use NFXP, CCP, or MPEC). |
| State marginal matching is the study question. | You want a feature-matching IRL baseline (use MaxEnt-IRL or MCE-IRL). |
| Transitions are known or pre-estimated. | The expert data is too sparse to estimate a reliable state marginal. |
| You want multiple f-divergence choices (forward KL, reverse KL, JS, chi-squared, TV). | You need standard errors on the recovered parameters. |
| An imitation or behavioral baseline with divergence control is needed. | Counterfactual re-solving in a structural model is the primary goal. |

## Quick Start

```python
from econirl.estimation import FIRLEstimator

estimator = FIRLEstimator(
    f_divergence="fkl",
    marginal_space="state",
    reward_scope="state",
)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.policy)
print(summary.metadata["occupancy_l1"])
```

## Evidence

f-IRL is reported on the paper-faithful state-marginal synthetic cell: 8 states,
3 actions, state-only reward, deterministic transitions, and a fully specified
data-generating process. The machine-readable results file records the reported
results and the action-dependent diagnostic cell. f-IRL also appears on the bus
engine and gridworld pages of the [simulation studies](../simulation_studies/index.md).

| Evidence | Current state |
| --- | --- |
| Evidence scope | Synthetic tabular simulation. |
| Primary cell | `f_irl_paper_state_marginal` (state marginal, state reward). |
| Machine-readable results file | [f_irl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/f_irl.json). |
| Counterfactual checks | Type A, Type B, and Type C are reported in the results file. |
| Action-dependent DDC | Diagnostic negative control; fails reward-range check. |

## f-IRL Guide

- [Context](f_irl/context.md)
- [Quick Start](f_irl/quick_start.md)
- [Under the Hood](f_irl/under_the_hood.md)
- [Pre-Estimation Checks](f_irl/pre_estimation.md)
- [Simulation Study](f_irl/validation.md)
- [Counterfactuals](f_irl/counterfactuals.md)

```{toctree}
:hidden:

f_irl/context
f_irl/quick_start
f_irl/under_the_hood
f_irl/pre_estimation
f_irl/validation
f_irl/counterfactuals
```
