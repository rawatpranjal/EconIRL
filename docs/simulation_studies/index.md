# Simulation Studies

Every page below is one experiment. We simulate data from a known model, run
the estimators on it, and report what they recover. Because the truth is
known, both recovery and failure are measurable.

| Page | Environment | Size | Estimators | What it shows |
| --- | --- | --- | --- | --- |
| [Bus engine replacement](rust_bus.md) | Keep-or-replace mileage model (Rust 1987). | 20 states x 2 actions | NFXP, CCP, TD-CCP, MCE-IRL, GLADIUS. | The canonical benchmark. Who recovers the cost parameters, and at what compute cost. |
| [Gridworld navigation](taxi_gridworld.md) | Walk to a goal on a grid. | 64 states x 5 actions | MCE-IRL, Neural MCE-IRL, AIRL, GLADIUS, NFXP, CCP. | What happens where the data rarely goes. |
| [Route choice](route_choice.md) | Synthetic road network (25 nodes, random geometric graph). | 25 states x 4 actions | NFXP, CCP, MCE-IRL, NeuralGLADIUS. | Structural parameter recovery and behavioral fidelity on a graph topology. |
| [Stockpiling](stockpiling.md) | Consumer stockpiling of a storable good (Hendel-Nevo). | 20 states x 2 actions | NFXP, CCP, MCE-IRL, NeuralGLADIUS. | Structural recovery on a price-driven inventory model where the optimal policy stockpiles on sale. |
| [Fleet maintenance](fleet_maintenance.md) | Multi-component bus engine replacement (Rust 1987, K=3 components). | 216 states x 2 actions | NFXP, CCP, MCE-IRL, NeuralGLADIUS. | High-dimensional factored environment. Structural recovery at scale; NeuralGLADIUS scalability on a factored state space. |
| [Content consumption](content_consumption.md) | Latent viewer types choosing what to watch (heterogeneous-agent model). | 65 states x 4 actions | AIRL-Het vs homogeneous AIRL and MCE-IRL. | Heterogeneity recovery. AIRL-Het sorts viewers into types and serves both, where a homogeneous fit settles on one and abandons the other. |

The findings in one line. On easy problems most estimators match the choice
probabilities. On harder problems many do not, and the gap in behavior is the
point. The differences also show up in parameter recovery, in counterfactuals,
and in compute cost.

## Reading the tables

All numbers come from a saved results file written by the run script.
Crashes and timeouts stay in the table with their error message.

Policy TV measures how far the estimated choice probabilities are from the
truth. Lower is better.

Regret measures welfare lost when the recovered model is used in a changed
environment. Type A shifts a payoff. Type B changes the dynamics. Type C
penalizes an action. Estimators that recover a transferable reward re-solve the
model and adapt. Policy-only methods keep their old policy, so their Type C
regret is large.

Parameter recovery is reported only for structural estimators. IRL methods
recover a reward that produces the same behavior but in a different
parameterization, so comparing their parameters to the truth is not
meaningful.

The estimators are documented in the [catalog](../estimators.md).

```{toctree}
:maxdepth: 1

rust_bus
taxi_gridworld
route_choice
stockpiling
fleet_maintenance
content_consumption
```
