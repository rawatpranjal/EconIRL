# Rust Bus Engine Example

## Important Links

- [MCE-IRL overview](../mce_irl.md)
- [Quick start](quick_start.md)
- [Pre-estimation checks](pre_estimation.md)
- [Bus engine simulation study](../../simulation_studies/rust_bus.md)

The bus replacement problem is a compact wiring example for MCE-IRL. It is not
a Ziebart paper replication. It also does not replace the Rust Table IX NFXP
replication.

Use an action-dependent reward matrix. One feature can assign mileage cost to
the keep action. A second can assign a fixed cost to replacement. The fitted
policy then gives replacement probabilities by mileage state.

When `transitions=None`, the wrapper estimates the keep-action
mileage-increment distribution from observations with action 0. It then
constructs the replacement kernel from the keep kernel's state-0 row. That
convenience is specific to the bus problem. A general MDP must supply either a
dense `(A, S, S)` transition tensor or `DeterministicTransitions`.

Interpret the fitted reward only under the supplied feature basis and
normalization. Use the bus engine simulation study for behavioral comparisons
across estimators.
