# Simulation Study

NNES runs on the `canonical_high_action` synthetic cell. The low-dimensional
`canonical_low_action` cell remains in the results file as a sanity check.
The simulation asks whether the NPL-profiled neural value path can recover a
finite-dimensional structural reward and counterfactual behavior when the state
representation is encoded and richer than the small tabular reference. Real
data cannot answer that question because the true reward, policy, value
function, Q function, and counterfactual objects are not observed.

These results are not hand-entered examples. They come from the simulation
harness. The harness fixes the transition law, reward features, and reward
weights before generating the finite panel. Those objects define the true
reward, policy, value function, Q function, and counterfactual targets held
back for evaluation. The estimator sees the generated panel, the transition
law, encoded state features, and reward features - not the oracle objects.

## Paper Simulation vs EconIRL Study

The original Nguyen simulation and the EconIRL study answer related but
different questions.

| Source | What it shows |
| --- | --- |
| Nguyen paper | NNES nearly matches oracle NFXP in Rust-style multi-module Monte Carlo designs where the oracle knows the separable structure and NNES does not. |
| EconIRL study | The package implementation recovers known reward, policy, value, Q, and counterfactual objects inside fixed low- and high-dimensional DGP cells. |

The paper simulation uses continuous mileage states, joint replacement actions,
50 buses, 20 kept periods after burn-in, 1,000 observations per replication,
and 100 replications in the two-module baseline. It also reports a
four-dimensional extension and high-discount anchoring checks.

The EconIRL results file is not a literal replication of that Monte Carlo. It is a
package simulation study: the DGP is held fixed, the oracle objects are known
before estimation, the high-dimensional primary cell has encoded states and 32
reward parameters, and every reported result is tied to
numerical checks.

The full result generator is
[`nnes_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nnes/run.py).
It writes the results file
[`nnes_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nnes.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/nnes/run.py --quiet-progress
```

## Read the Tables

Read the tables as a sequence. The design rows state which synthetic cells
were run. The fit summary reports convergence, L-BFGS-B iterations, final
value-network loss, likelihood, and run time. Recovery metrics compare the
estimated structural object to oracle reward, policy, value, and Q objects.
Numerical checks are the numerical cutoffs recorded by the harness.

The simulation result does not rely on optimizer convergence alone. It also
uses value-network loss, recovery against true simulated values, and counterfactual regret.

## Design

| Quantity | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Total states | 21 | 81 |
| State dimension | 2 | 16 |
| Actions | 3 | 3 |
| Reward parameters | 4 | 32 |
| Simulated individuals | 2,000 | 2,000 |
| Periods per individual | 80 | 80 |
| Observations | 160,000 | 160,000 |
| Feature rank | 4 / 4 | 32 / 32 |
| State-action coverage | 1.000 | 0.959 |

## Fit Summary

| Quantity | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Converged | true | true |
| L-BFGS-B iterations | 10 | 29 |
| Log likelihood | -174875.8093 | -160272.8803 |
| Estimation time | 37.12 seconds | 36.46 seconds |
| Outer NPL iterations | 3 | 3 |
| Final V loss | 0.000399 | 0.029932 |

The primary high-dimensional cell ran in 36.46 seconds.

## Recovery Metrics

| Metric | Threshold | Low-dimensional | High-dimensional primary |
| --- | --- | ---: | ---: |
| Parameter cosine | at least 0.95 | 0.998240 | 0.991204 |
| Parameter relative RMSE | at most 0.30 | 0.065179 | 0.135110 |
| Reward RMSE | at most 0.08 | 0.010210 | 0.064012 |
| Policy total variation | at most 0.03 | 0.005646 | 0.023834 |
| Value RMSE | at most 0.20 | 0.019845 | 0.115620 |
| Q RMSE | at most 0.20 | 0.023370 | 0.137145 |

## Numerical Checks

| Check | Threshold | Primary value | Status |
| --- | --- | ---: | --- |
| Outer NPL iterations | at least 3 | 3 | pass |
| Final V loss | at most 0.05 | 0.029932 | pass |
| Parameter cosine | at least 0.95 | 0.991204 | pass |
| Parameter relative RMSE | at most 0.30 | 0.135110 | pass |
| Reward RMSE | at most 0.08 | 0.064012 | pass |
| Policy total variation | at most 0.03 | 0.023834 | pass |
| Value RMSE | at most 0.20 | 0.115620 | pass |
| Q RMSE | at most 0.20 | 0.137145 | pass |
| Type A regret | at most 0.05 | 0.004865 | pass |
| Type B regret | at most 0.05 | 0.005559 | pass |
| Type C regret | at most 0.05 | 0.001314 | pass |

## Counterfactual Recovery

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.021578 | 0.002165 | 0.007880 | 0.004865 |
| Type B | 0.021651 | 0.002128 | 0.007916 | 0.005559 |
| Type C | 0.013053 | 0.000920 | 0.002325 | 0.001314 |

The estimates are not exactly equal to the true parameters because the panel
is finite and the value function is approximated. The study reports recovery
within the listed tolerances in the synthetic cells.

NNES also appears in the cross-estimator comparison on the
[bus engine replacement](../../simulation_studies/rust_bus.md) simulation
study page, where it is benchmarked against the full structural and IRL
rosters on a synthetic bus engine panel.
