# Simulation Study

## Important Links

- [AIRL Overview](../airl.md)
- [Controlled recovery results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl_controlled_recovery.json)
- [Bootstrap results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl_bootstrap_calibration.json)
- [Taxi Dynamics Transfer](taxi_transfer.md)

AIRL is graded on three separate questions. The controlled study asks whether
the adversarial estimator recovers a state reward and its behavior. The
bootstrap study asks whether trajectory intervals cover reward and policy
functionals. The taxi study asks whether the reward can be re-solved after a
material traffic change.

## Controlled recovery

The design has 16 states, 4 actions, 4 state features, 300 individuals, and 80
periods. Three independent panels and training seeds were used.

| Metric | Median | 95th percentile |
| --- | ---: | ---: |
| Reward normalized RMSE | 0.1397 | 0.1422 |
| Policy total variation | 0.0064 | 0.0083 |
| Value normalized RMSE | 0.1521 | 0.1523 |
| Q normalized RMSE | 0.1642 | 0.1646 |
| Transfer policy TV | 0.0067 | 0.0101 |
| Transfer regret | 0.0040 | 0.0070 |

All three fits converged. State-action coverage was 1.000 in every panel.

## Bootstrap calibration

The calibration design uses a six-state decomposable MDP with a state-only
linear reward. Each of 20 panels contains 160 individual trajectories of 40
periods. Each fit uses 19 whole-trajectory resamples.

| Functional family | Coverage | Lower miss | Upper miss | Median width |
| --- | ---: | ---: | ---: | ---: |
| Centered reward | 0.900 | 0.050 | 0.050 | 0.0927 |
| Policy probability | 0.900 | 0.050 | 0.050 | 0.0179 |

All 380 resampled fits succeeded. Mean absolute bias was 0.0261 for centered
reward cells and 0.0060 for policy probabilities.

## Paper boundary

These studies use the adversarial estimator and paper-supported conditions.
They do not reproduce a published AIRL result table. Fu et al. Section 7.1
starts from MaxEnt IRL. Calling that experiment an exact adversarial AIRL
replication would conflate two algorithms.

## Reproduce

Run the combined gate after the four result files and applied notebook have
been generated.

```bash
PYTHONPATH=src:. uv run python validation/estimators/airl/qualification_report.py
```

**Result**

```text
AIRL qualification
==================
Controlled recovery: 3/3 converged, reward NRMSE median=0.139664, policy TV p95=0.008312
Controlled transfer: policy TV p95=0.010145, regret p95=0.007025
Trajectory bootstrap: 20/20 usable panels, 380/380 successful draws
Bootstrap coverage: reward=0.900, policy=0.900
Taxi transfer: oracle change TV=0.109485, fitted transfer TV p95=0.052501, flow regret p95=0.007120
Serialization: fresh wheel process with exact supported-output parity
Notebook: all cells executed from the installed wheel with no errors
Paper boundary: generated adversarial recovery and transfer studies, not an exact replication of Fu et al. Section 7.1
```
