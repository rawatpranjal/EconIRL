# Simulation Studies

Every page below is one experiment. We simulate data from a known model, run
the estimators on it, and report what they recover. Because the truth is
known, both recovery and failure are measurable.

| Page | Environment | Size | Estimators | What it shows |
| --- | --- | --- | --- | --- |
| [Bus engine replacement](rust_bus.md) | Keep-or-replace mileage model (Rust 1987). | 20 states x 2 actions | All. | The canonical benchmark. Who recovers the cost parameters, and at what compute cost. |
| [Gridworld navigation](taxi_gridworld.md) | Walk to a goal on a grid. | 64 states x 5 actions | All, IRL focus. | What happens where the data rarely goes. |
| [Direct optimization](direct_optimization.md) | Estimation under correct and misspecified rewards. | varies | MPEC, neural MPEC, GLADIUS. | How this family degrades under reward misspecification. |
| [Route choice](route_choice.md) | Synthetic road network (25 nodes, random geometric graph). | 25 states x 4 actions | Structural (NFXP, CCP, MPEC), MCE-IRL, NeuralGLADIUS. | Structural parameter recovery and behavioral fidelity on a graph topology. |
| [High-dimensional route choice](highdim_route_choice.md) | Synthetic road network at scale (150 nodes). | 150 states x 4 actions | RHIP horizon sweep (H=0/1/3/inf), MCE-IRL, AIRL, structural. | The RHIP planning-horizon spectrum at scale. H=inf matches MCE-IRL; accuracy improves along the horizon. |
| [RHIP horizon recovery](rhip_lookahead.md) | Route choice with finite-lookahead demonstrators (25 nodes). | 25 states x 4 actions | RHIP horizon sweep (H=0/1/2/3/5/inf). | The best-fitting planning horizon recovers the demonstrator's true lookahead. An interior horizon beats both Max-Margin Planning and Max Causal Entropy, and the optimum shifts with the demonstrator. |
| [Stockpiling](stockpiling.md) | Consumer stockpiling of a storable good (Hendel-Nevo). | 20 states x 2 actions | Structural (NFXP, CCP, MPEC), MCE-IRL, NeuralGLADIUS. | Structural recovery on a price-driven inventory model where the optimal policy stockpiles on sale. |
| [Fleet maintenance](fleet_maintenance.md) | Multi-component bus engine replacement (Rust 1987, K=3 components). | 216 states x 2 actions | Structural (NFXP, CCP, MPEC), MCE-IRL, NeuralGLADIUS. | High-dimensional factored environment. Structural recovery at scale; NeuralGLADIUS scalability on a factored state space. |
| [Vehicle scrappage](vehicle_scrappage.md) | Optimal vehicle replacement using Dutch RDW inspection data (Rust 1987 framework). | 75 states x 2 actions | Structural (NFXP, CCP, MPEC, UFXP, NNES), MCE-IRL. | Classical vs modern structural estimators on a realistic optimal stopping problem. |
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
direct_optimization
route_choice
highdim_route_choice
rhip_lookahead
stockpiling
fleet_maintenance
vehicle_scrappage
content_consumption
```
