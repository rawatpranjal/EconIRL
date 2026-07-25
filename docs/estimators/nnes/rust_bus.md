# Rust Bus Engine Example

Read this page as a package smoke test. It is intentionally smaller than the
paper simulations and does not replace the NNES validation page.

The Rust bus-engine replacement problem is the canonical dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep
the current engine or replace it.

NNES can be used through the dataframe wrapper for a quick package smoke test.
This page is intentionally smaller than Nguyen's original simulation, which
uses multi-module continuous mileage states and compares NNES with an oracle
NFXP benchmark.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import NNES

df = load_rust_bus()

model = NNES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility=rust_bus_reward_spec(90),
    bellman="npl",
    hidden_dim=32,
    num_layers=2,
    v_epochs=500,
    n_outer_iterations=3,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.predict_proba([0, 20, 40, 60, 80])[:, 1])
```

Output from the package smoke run:

```text
{'theta_c': 0.001034, 'RC': 3.073617}
[0.044209 0.060566 0.081061 0.103800 0.114049]
```

## Interpretation

The Rust-style linear utility specification estimates operating-cost and
replacement-cost parameters. The fitted policy gives the replacement
probability by mileage bin. The fitted `v_network_` vector stores neural
value-network predictions for the discrete states used by the wrapper.

## Full API Variant

Use `econirl.estimation.nnes.NNESEstimator` for package-native `Panel`
objects, custom reward features, supplied transition tensors, or exact control
over neural value training.

```python
from econirl.estimation.nnes import NNESConfig, NNESEstimator

estimator = NNESEstimator(
    config=NNESConfig(
        hidden_dim=32,
        num_layers=2,
        v_epochs=500,
        n_outer_iterations=3,
        compute_se=True,
    )
)
summary = estimator.estimate(panel, utility, problem, transitions)

print(summary.parameters)
print(summary.metadata["v_loss_per_outer"][-1])
```

No fixed output is shown for the estimator-API snippet because `panel`,
`utility`, `problem`, and `transitions` are caller-supplied objects.

## Replication Boundary

This page is a package smoke example, not the full historical Rust
replication and not the full Nguyen Monte Carlo. The reported simulation study
uses a simulation results file, where reward, policy, value, Q, and
counterfactual oracle objects are available for comparison. See
[Simulation Study](validation.md) for that path.
