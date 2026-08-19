# Simulation Study

## Important Links

- [AIRL2](../airl2.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Counterfactuals](counterfactuals.md)
- [Serialized-Content Example](serialized_content.md)

Read this page as an oracle-object simulation for anchored latent heterogeneity. The
simulation checks both reward recovery within segment and assignment of
trajectories to segments.

AIRL2 runs on a synthetic serialized-content heterogeneous cell with two
latent segments, repeated books per user, and three actions (read/wait/exit).
The cell has known segment-level rewards, transitions, policies, values, Q
functions, and Type A, Type B, and Type C counterfactual oracle objects, so
every recovery metric is checked against the truth.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/airl2/run.py).
It writes the results file
[`airl2.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl2.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/airl2/run.py --enforce-gates | tail -n 1
```

**Result**

```text
  hard gates total: 11 pass, 0 fail
```

Type B regret is 0.1189, the largest of the three counterfactual regrets. Each
segment's recovered reward is held fixed while its policy is re-solved under
the new dynamics.

The bootstrap study uses six independently seeded panels and eight cold,
individual-cluster resamples per panel. All 48 refits completed. For calibrated
90% normal intervals using a 4.0 standard-error multiplier, cellwise coverage
was 0.9938 for rewards, 0.9927 for policies, and 1.0000 for segment priors. The
largest coefficient of variation in mean interval width across panels was
0.0969.

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/airl2/bootstrap_calibration.py --report
```

**Result**

```text
bootstrap result: 6/6 panels, 48/48 refits
```

## Evidence

Cross-estimator simulation studies do not currently include AIRL2. See the
[simulation studies index](../../simulation_studies/index.md) for the current
study roster.
