# Rust bus engine replacement

The canonical structural benchmark: Harold Zurcher's bus engine replacement
problem (Rust 1987). A single-agent, binary-action (keep or replace) decision
over a discretized mileage state. This experiment runs as many estimators as can
fit the problem on the same simulated panel, and reports the exact recovered
operating and replacement costs, policy distance, and counterfactual regret.

Environment: `RustBusEnvironment` (20 mileage bins, operating cost 0.01,
replacement cost 2.0, discount 0.95). Roster: the full set that accepts the
uniform estimate interface.

```{note}
Status: results pending. This page is generated from
`validation/results/sim_rust_bus.json` once the run completes.
```
