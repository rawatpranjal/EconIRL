# Simulation Study

AIRL-Het runs on a synthetic serialized-content heterogeneous cell with two
latent segments, repeated books per user, and three actions (read/wait/exit).
The cell has known segment-level rewards, transitions, policies, values, Q
functions, and Type A, Type B, and Type C counterfactual oracle objects, so
every recovery metric is checked against the truth.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/aairl/run.py).
It writes the results file
[`aairl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/aairl/run.py
```

The Type B regret is close to its threshold. This is expected for the
heterogeneous transition-change counterfactual: each segment's reward must
adapt to a different new dynamic, and the recovered reward carries more
approximation error under that intervention than under reward shifts or action
removals.

## Evidence

Cross-estimator simulation-study coverage for AIRL-Het is planned. See the
[simulation studies index](../../simulation_studies/index.md) for the current
study roster.
