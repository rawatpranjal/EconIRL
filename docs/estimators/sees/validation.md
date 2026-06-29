# Simulation Study

The SEES simulation study uses the `canonical_high_action` synthetic cell.
The low-dimensional `canonical_low_action` cell remains in the results file as
a sanity check for the historical state-index basis. The simulation asks
whether a deterministic sieve value approximation can recover structural reward
and counterfactual behavior when the state representation is encoded and richer
than a small tabular state index. Real data cannot answer that question because
the true reward, policy, value function, Q function, and counterfactual oracles
are not observed.

These results come from the simulation harness. The harness fixes the reward,
transition law, optimal policy, value function, Q function, and counterfactual
oracle objects before generating the finite panel. The estimator sees the
generated panel, the transition law, and the encoded reward basis, not the
oracle dynamic objects.

The full result generator is
[`sees_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/sees/run.py).
It writes the results file
[`sees_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/sees.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/sees/run.py --quiet-progress
```

The estimates are not exactly equal to truth because the panel is finite.

## Exact Rust Bus Oracle Check

The component test suite also includes a smaller Rust bus oracle check with a
full B-spline basis. At the true Rust parameters, `solution="value"`,
`solution="q"`, `solution="ev"`, `solution="policy"`, and
`solution="collocation"` match the oracle policy, value function, and
Q function with Bellman residual below `1e-8`.

That is an oracle representability and equilibrium-residual check, not evidence
that a finite stochastic panel produces bit-for-bit parameter equality.

## Random-Start Rust Bus Recovery

The slow benchmark checks practical finite-sample recovery from random
structural starts. It simulates the same Rust bus panel, fits NFXP as the
finite-sample structural reference, then runs all five SEES solution modes
from random initial parameters with `num_theta_starts=4`.

The check compares SEES to the NFXP estimate on the same panel. It does not
compare directly to the population truth, because finite stochastic panels do
not make the sample MLE exactly equal to the DGP parameters.

```bash
pytest tests/benchmarks/test_parameter_recovery.py::test_sees_rust_bus_solution_variants
```

Results file:
[`sees_rust_random_start.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/sees_rust_random_start.json).

## Evidence

SEES appears on the
[bus engine](../../simulation_studies/rust_bus.md) page. See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
