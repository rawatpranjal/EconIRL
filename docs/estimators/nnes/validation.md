# Simulation Study

NNES runs on the `canonical_high_action` synthetic cell. The low-dimensional
`canonical_low_action` cell remains in the results file as a sanity check.
The simulation asks whether the NPL-profiled neural value path can recover a
finite-dimensional structural reward and counterfactual behavior when the state
representation is encoded and richer than the small tabular reference. Real
data cannot answer that question because the true reward, policy, value
function, Q function, and counterfactual objects are not observed.

These results come from the simulation harness. The harness fixes the
transition law, reward features, and reward weights before generating the
finite panel. Those objects define the true reward, policy, value function,
Q function, and counterfactual targets held back for evaluation. The estimator
sees the generated panel, the transition law, encoded state features, and
reward features, not the oracle objects.

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

The EconIRL results file is not a literal replication of that Monte Carlo. It
is a package simulation study: the DGP is held fixed, the oracle objects are
known before estimation, the high-dimensional primary cell has encoded states
and 32 reward parameters, and every reported result is tied to numerical
checks.

The full result generator is
[`nnes_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nnes/run.py).
It writes the results file
[`nnes_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nnes.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/nnes/run.py --quiet-progress
```

The estimates are not exactly equal to the true parameters because the panel
is finite and the value function is approximated.

## Evidence

NNES appears in two simulation studies:
[bus engine](../../simulation_studies/rust_bus.md) and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
