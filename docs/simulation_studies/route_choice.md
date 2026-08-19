# Route choice on a synthetic road network

A traveller moves through a road network one step at a time. Each period, the agent chooses among the nearest neighbours of the current node. The utility depends on the edge: how long it is, how attractive the destination is, and how close the destination sits to a fixed goal node.

![An aerial photograph of the West End of New Orleans in 1941, with a highway running along the lakeshore into the distance.](../_static/simulation_studies/route_choice_photo.jpg)

*West End, New Orleans, with the Old Hammond Highway along Lake Pontchartrain, 1941. Public domain, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:West_End_New_Orleans_and_Old_Hammond_Highway,_1941.jpg).*

## The data-generating process

Nodes are scattered uniformly at random in the unit square. Edges connect pairs within a fixed Euclidean radius. A spanning tree is overlaid to keep the graph connected. The agent at node $s$ picks among $A$ nearest neighbours sorted by distance. Actions beyond the node degree self-loop.

The reward for traversing edge $(s, a) \to s'$ is linear in three features:

$$
u_\theta(s, a) = \theta_{\mathrm{cost}}\,(-d_{ss'}) + \theta_{\mathrm{am}}\,\mathrm{am}(s') + \theta_{\mathrm{goal}}\,(-\ell_{s'})
$$

where $d_{ss'}$ is the Euclidean edge length, $\mathrm{am}(s')$ is a node-level amenity draw, and $\ell_{s'}$ is the shortest-path distance from $s'$ to a fixed destination node. The true parameters are $\theta = [1.0,\;0.5,\;1.0]$.

Agents discount future payoffs at $\beta$ and face i.i.d. logit taste shocks (scale $\sigma = 1$). Their behaviour solves the soft Bellman equation. All three parameters are identified from observed route choices because the features vary across edges, not just states. The panel simulates $N$ agents for $T$ periods from the true optimal policy. The figure shows simulated paths and the optimal value function (lower at nodes far from the goal).

Synthetic route choice on a random geometric graph. ``road_network(num_nodes=25, num_actions=4, discount_factor=0.95, seed=0)``. 200 x 35 observations, 30 replications, seed 42. True theta `[1.0, 0.5, 1.0]`. Design rank 3/3, condition number 1.34e+01, action-contrast rank 3/3 (the rank that identification from choices actually uses). Generated 2026-06-28 with econirl 0.0.8.

![Simulated trajectories and the optimal value function for Route choice (25 nodes, 4 actions)](../_static/simulation_studies/route_choice_dgp.png)

## Estimators and data

| Estimator | Family | Uses transitions $P(s'\mid s,a)$ | Transferable reward | Standard errors |
|---|---|---|---|---|
| NFXP | structural | yes | yes | yes |
| CCP | structural | yes | yes | yes |
| MCE-IRL | behavioral | yes | yes | no |

Uses transitions is whether the estimator reads the transition kernel; model-free learners do not. Transferable reward is whether it recovers a reward that re-solves under a counterfactual. Standard errors is whether it returns inference. The last two are read from the run.

## Results

| Estimator | Family | Ran | Conv | Recovered params | Param RMSE | Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NFXP | structural | 30/30 | 30/30 | [1.024, 0.495, 0.990] | 0.0733 | 0.0081 | 0.0034 | 0.0028 | 0.0068 | 0.0050 | 0.9 |
| CCP | structural | 30/30 | 1/30 | [1.024, 0.495, 0.990] | 0.0733 | 0.0081 | 0.0034 | 0.0028 | 0.0068 | 0.0050 | 3.5 |
| MCE-IRL | behavioral | 30/30 | 2/30 | [1.024, 0.495, 0.990] | - | 0.0081 | 0.0034 | 0.0028 | 0.0068 | 0.0050 | 3.5 |

Param RMSE covers the structural family only, which shares the parameterization of the true model. Policy TV is the distance between estimated and true choice probabilities, lower is better. Conv is the estimator's own convergence indicator. A cautious estimator can report False while the recovered policy is accurate. Regret base is welfare lost in the observed environment. Types A, B, and C are welfare lost after a change. Type A shifts a payoff, Type B changes the transitions, Type C penalizes an action. Estimators with a recovered reward re-solve it and adapt. Those without one keep their old policy.

![Policy total variation per estimator for Route choice (25 nodes, 4 actions)](../_static/simulation_studies/route_choice_results.png)

## Parameter recovery

| Estimator | Parameter | True | Mean est | Bias | Emp. SE | RMSE | 95% coverage | SE avail |
|---|---|---|---|---|---|---|---|---|
| NFXP | edge_cost | 1.000 | 1.024 | +0.024 | 0.139 | 0.139 | 1.00 +/- 0.00 | 100% (30 reps) |
| NFXP | amenity | 0.500 | 0.495 | -0.005 | 0.021 | 0.021 | 1.00 +/- 0.00 | 100% (30 reps) |
| NFXP | goal | 1.000 | 0.990 | -0.010 | 0.042 | 0.043 | 1.00 +/- 0.00 | 100% (30 reps) |
| CCP | edge_cost | 1.000 | 1.024 | +0.024 | 0.139 | 0.139 | - | 17% (30 reps) |
| CCP | amenity | 0.500 | 0.495 | -0.005 | 0.021 | 0.021 | - | 17% (30 reps) |
| CCP | goal | 1.000 | 0.990 | -0.010 | 0.042 | 0.043 | - | 17% (30 reps) |

Coverage is the share of replications whose 95% interval contains the truth, shown with its Monte Carlo standard error. It is computed only where every replication produced a finite standard error. SE avail is the share of replications with finite standard errors.

The structural family (NFXP, CCP) recovers all three parameters on the same scale as the truth, so Param RMSE applies to them alone. MCE-IRL here uses the same linear features and recovers the same values, but its weights stay out of the recovery table because an IRL reward is only partially identified in general. Policy TV and regret are the behavioral scorecards.

## Scaling

The same study at three problem sizes (15, 25, 40 states). Each line is one estimator: fit time on the left, policy total variation on the right. The structural methods stay accurate across sizes. The fits run from sub-second to a few seconds, so the compute lines reflect fixed overhead more than asymptotics at this scale.

![Fit time and policy total variation against the number of states](../_static/simulation_studies/route_choice_scaling.png)

## Reward and structure

The network sits in the unit square. Each dot is a node, each line an edge. The color is the optimal value $V^*(s)$ at the true parameters: value rises toward the goal node (the star). The spatial structure is the routing problem.

![Road network layout colored by optimal value](../_static/simulation_studies/route_choice_network.png)

The recovered reward sits beside the true reward as state-by-action heatmaps, sharing one color scale. The recovered panel uses the best structural estimator's mean coefficients. The two panels read the same, so the linear reward is recovered across the state space.

![True and recovered reward heatmaps](../_static/simulation_studies/route_choice_reward.png)

## Notes per estimator

**NFXP.** Full-solution MLE with a nested Bellman fixed-point inner loop. Quadratic convergence near the optimum; all three parameters are identified from the edge-feature contrast.

**CCP.** CCP uses a first-step nonparametric policy estimate to avoid the inner Bellman loop. One policy-iteration step corrects the bias from the nonparametric first stage.

**MCE-IRL.** Its convergence indicator reports whether the gradient norm crossed the tolerance. The objective often plateaus before that, so it can read False while the recovered policy is accurate.

## Reproduce

```bash
python scripts/study_route_choice.py                 # run + write JSON
python scripts/study_route_choice.py --page          # regenerate this page
python scripts/study_route_choice.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/study_route_choice.json`.

Not shown on this page: MPEC (an Other-tier constrained-optimization form of the same MLE; NFXP and CCP carry the structural recovery here); IQ-Learn, f-IRL (not separately identified from choices on this problem; reward is only partially identified from behavior); NNES, SEES, TD-CCP, UFXP (correct structural estimators but slower on a 25-node graph; NFXP and CCP already cover the structural family); MaxEnt-IRL, MaxMargin-IRL, AIRL, Deep-MCE-IRL (trajectory-entropy and max-margin objectives are not the choice model that generated the data; AIRL also requires a state-only reward and cannot represent the edge contrast); NeuralGLADIUS (a model-free neural policy learner; on a 25-node tabular state space it reduces to behavioral cloning on the choice likelihood with no structural edge, and it trailed every method shown. Its arena is scalability and predictive likelihood on large state spaces where full-solution methods become intractable, a regime outside these policy-recovery studies).