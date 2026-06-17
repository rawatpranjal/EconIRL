# Simulation Studies

Every page below is one experiment. We simulate data from a known model, run
the estimators on it, and report what they recover. Because the truth is
known, both recovery and failure are measurable.

| Page | Environment | Size | Estimators | What it shows |
| --- | --- | --- | --- | --- |
| [Bus engine replacement](rust_bus.md) | Keep-or-replace mileage model (Rust 1987). | 20 states x 2 actions | All. | The canonical benchmark. Who recovers the cost parameters, and at what compute cost. |
| [Gridworld navigation](taxi_gridworld.md) | Walk to a goal on a grid. | 64 states x 5 actions | All, IRL focus. | What happens where the data rarely goes. |
| [Abstract MDP 1](abstract_mdp_1_sanity.md) | Small random MDP, linear reward. | 8 states x 2 actions | All. | An easy problem every correct estimator must pass. |
| [Abstract MDP 2](abstract_mdp_2_harder.md) | The same generator, hardened three ways. | 300 states; 24-state collinear cell | Structural family. | Runtime at scale, inference near discount one, and broken identification. |
| [Abstract MDP 3: High Dimensional Case](abstract_mdp_3_highdim.md) | The same generator at large scale. | 3000 states x 2 actions | Ten estimators across families. | How compute costs separate as the state space grows. |
| [Abstract MDP 4: Interaction effect](abstract_mdp_4_nonlinear.md) | A reward that multiplies two features the estimators do not model. | 24 states x 3 actions | All. | What an omitted interaction costs: a small behavioral miss, a larger counterfactual one. |
| [Direct optimization](direct_optimization.md) | Estimation under correct and misspecified rewards. | varies | MPEC, neural MPEC, GLADIUS. | How this family degrades under reward misspecification. |
| [Route choice](route_choice.md) | Synthetic road network (25 nodes, random geometric graph). | 25 states x 4 actions | Structural (NFXP, CCP, MPEC), MCE-IRL, NeuralGLADIUS. | Structural parameter recovery and behavioral fidelity on a graph topology. |
| [Stockpiling](stockpiling.md) | Consumer stockpiling of a storable good (Hendel-Nevo). | 20 states x 2 actions | Structural (NFXP, CCP, MPEC), MCE-IRL, NeuralGLADIUS. | Structural recovery on a price-driven inventory model where the optimal policy stockpiles on sale. |
| [Content consumption](content_consumption.md) | Serialized content with two hidden reader segments (Lee-Sudhir-Wang). | 61 states x 3 actions | AIRL-Het, vs pooled AIRL, BC, NFXP, CCP. | Recovering latent segments: AIRL-Het finds both reader types; single-type models match average behaviour and miss them. |

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
abstract_mdp_1_sanity
abstract_mdp_2_harder
abstract_mdp_3_highdim
abstract_mdp_4_nonlinear
direct_optimization
route_choice
stockpiling
content_consumption
```
