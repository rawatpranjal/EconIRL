# Simulation Study

f-IRL is reported on two synthetic cells. The primary cell is the source-paper
state-marginal configuration: a fully specified data-generating process with
known reward, transitions, policy, value, Q function, and Type A, B, and C
counterfactual oracle objects. The action-dependent cell is a diagnostic
negative control that fails the reward-range check and is not treated as
structural recovery evidence.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/f_irl/run.py).
It writes the results file
[`f_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/f_irl.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/f_irl/run.py
```

The action-dependent cell (`canonical_low_action`) fails the reward-range check
with a reward range of 0.000, indicating a flat-reward solution. This cell is
retained as a diagnostic to show that f-IRL with action-dependent marginal
matching does not recover a non-trivial reward on the standard DDC benchmark.
It is not structural recovery evidence.

## Evidence

f-IRL appears on the
[taxi gridworld](../../simulation_studies/taxi_gridworld.md) page. See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
