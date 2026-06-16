# Simulation Study

The SEES simulation study uses the `canonical_high_action` synthetic cell.
The low-dimensional `canonical_low_action` cell remains in the results file as a
sanity check for the historical state-index basis. The simulation asks whether
a deterministic sieve value approximation can recover structural reward and
counterfactual behavior when the state representation is encoded and richer
than a small tabular state index. Real data cannot answer that question because
the true reward, policy, value function, Q function, and counterfactual oracles
are not observed.

These results are not hand-entered examples. They come from the simulation
harness. The harness fixes the reward, transition law, optimal policy, value
function, Q function, and counterfactual oracle objects before generating the
finite panel. The estimator sees the generated panel, the transition law, and
the encoded reward basis, not the oracle dynamic objects.

The full result generator is
[`sees_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/sees/run.py).
It writes the results file
[`sees_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/sees.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/sees/run.py --quiet-progress
```

## Read the Tables

Read the tables as a sequence. The design rows state which synthetic cells
were run. The fit summary reports the optimizer flag, Bellman residuals,
likelihood, and run time. Recovery metrics compare the estimated structural
object to oracle reward, policy, value, and Q objects. Numerical checks are
the reported thresholds.

The optimizer flag is reported exactly as returned by L-BFGS-B. The reported
scope does not rely on the optimizer flag alone; it also uses the Bellman
residual, finite standard errors, recovery against true simulated values, and counterfactual
regret.

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
| Basis source | `state_index` | `encoded_state` |
| Basis dimension | 21 | 81 |
| Penalty weight | 100 | 10000 |

## Fit Summary

| Quantity | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Optimizer flag | false | false |
| Iterations | 181 | 6 |
| Log likelihood | -174875.7843 | -160289.8089 |
| Estimation time | 8.27 seconds | 6.01 seconds |
| Bellman violation | 5.83e-5 | 3.08e-6 |
| Bellman RMSE | 2.74e-5 | 6.80e-7 |
| Standard errors finite | true | true |

The primary high-dimensional cell ran in 6.01 seconds.

## Recovery Metrics

| Metric | Gate | Low-dimensional | High-dimensional primary |
| --- | --- | ---: | ---: |
| Parameter cosine | at least 0.99 | 0.999146 | 0.999955 |
| Parameter relative RMSE | at most 0.15 | 0.059671 | 0.009528 |
| Reward RMSE | at most 0.03 | 0.008988 | 0.004432 |
| Policy total variation | at most 0.02 | 0.005179 | 0.002117 |
| Value RMSE | at most 0.10 | 0.017591 | 0.037836 |
| Q RMSE | at most 0.10 | 0.020514 | 0.031480 |

## Numerical Checks

| Gate | Threshold | Primary value | Status |
| --- | --- | ---: | --- |
| Bellman violation | at most 0.05 | 0.000003 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | at least 0.99 | 0.999955 | pass |
| Parameter relative RMSE | at most 0.15 | 0.009528 | pass |
| Reward RMSE | at most 0.03 | 0.004432 | pass |
| Policy total variation | at most 0.02 | 0.002117 | pass |
| Value RMSE | at most 0.10 | 0.037836 | pass |
| Q RMSE | at most 0.10 | 0.031480 | pass |
| Type A regret | at most 0.01 | 0.000113 | pass |
| Type B regret | at most 0.01 | 0.000183 | pass |
| Type C regret | at most 0.01 | 0.000014 | pass |

## Counterfactual Recovery

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.002096 | 2.78e-5 | 0.000234 | 0.000113 |
| Type B | 0.002421 | 3.67e-5 | 0.000309 | 0.000183 |
| Type C | 0.001231 | 6.91e-6 | 0.000027 | 0.000014 |

The estimates are not exactly equal to truth because the panel is finite. The
reported scope is recovery within the listed tolerances in the frozen
synthetic cells.

## Exact Rust Bus Oracle Check

The component test suite also includes a smaller Rust bus oracle check with a
full B-spline basis. At the true Rust parameters, `solution="value"`,
`solution="q"`, `solution="ev"`, `solution="policy"`, and
`solution="collocation"` match the oracle policy, value function, and
Q function with Bellman residual below `1e-8`.

That is an oracle representability and equilibrium-residual check, not evidence
that a finite stochastic panel produces bit-for-bit parameter equality.

## Random-Start Rust Bus Recovery

The slow benchmark also checks practical finite-sample recovery from random
structural starts. It simulates the same Rust bus panel, fits NFXP as the
finite-sample structural reference, then runs all five SEES solution modes
from random initial parameters with `num_theta_starts=4`.

The check compares SEES to the NFXP estimate on the same panel. It does not
compare directly to the population truth, because finite stochastic panels do
not make the sample MLE exactly equal to the DGP parameters.

```bash
pytest tests/benchmarks/test_parameter_recovery.py::test_sees_rust_bus_solution_variants
```

Generated results files:

- - [`sees_rust_random_start.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/sees_rust_random_start.json)

| Mode | Gate count | Max param RMSE | Max policy TV | Max value RMSE | Max Q RMSE | Max Bellman | Max grad norm | Optimizer flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `value` | 9/9 | 0.000859 | 0.000046 | 0.005762 | 0.005820 | 3.365e-04 | 1.062e-03 | 0/9 |
| `q` | 9/9 | 0.001960 | 0.000109 | 0.015739 | 0.015420 | 9.297e-04 | 8.793e-04 | 0/9 |
| `ev` | 9/9 | 0.002212 | 0.000082 | 0.014964 | 0.014944 | 8.389e-04 | 6.783e-04 | 0/9 |
| `policy` | 9/9 | 0.006692 | 0.000200 | 0.031827 | 0.033279 | 7.661e-04 | 5.564e-03 | 0/9 |
| `collocation` | 9/9 | 0.000859 | 0.000046 | 0.005762 | 0.005820 | 3.365e-04 | 1.062e-03 | 0/9 |

The optimizer flag is the JAXopt gradient flag. The recovery check is
based on the finite-sample recovery metrics and Bellman residuals above.
