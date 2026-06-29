# Simulation Study

Read this page as an oracle-object simulation for the planning-horizon behavior.
The study asks whether the chosen horizon matches the demonstrator's planning
depth when the reward and transitions are known.

RHIP runs on two synthetic studies. The primary study is the
high-dimensional route-choice problem on a 150-node random geometric road
network. The second study fixes a known demonstrator lookahead and sweeps
the estimator horizon to show that the best-fitting horizon tracks the
demonstrator's. Together they check whether the horizon spectrum behaves
as the theory predicts when the transition law, reward-feature basis, and
reward weights are all known before generating the panel. Real data cannot
answer that question because the reward, policy, value function, and
counterfactual oracles are not observed.

The numbers come from the simulation harness. In that harness, the
transition law, reward features, and reward weights are fixed before
generating the panel. The estimator sees the generated demonstrations, the
transition law, and the supplied reward features. The reward, policy, value
function, and counterfactual oracles are held back for evaluation.

The full result generator for the primary study is
[`scripts/study_highdim_route_choice.py`](https://github.com/rawatpranjal/EconIRL/blob/main/scripts/study_highdim_route_choice.py).
It writes the results file
[`study_highdim_route_choice.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/study_highdim_route_choice.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python scripts/study_highdim_route_choice.py
```

The lookahead recovery study is
[`scripts/study_rhip_lookahead.py`](https://github.com/rawatpranjal/EconIRL/blob/main/scripts/study_rhip_lookahead.py):

```bash
PYTHONPATH=src:. python scripts/study_rhip_lookahead.py
```

## Evidence

### Horizon spectrum

The primary study sweeps $H \in \{0, 1, 3, \infty\}$ on a 150-node random
geometric road network with true parameters $\theta = [1.0, 0.5, 1.0]$ over
features `[edge_cost, amenity, goal]`. Policy total variation falls monotonically
as the horizon grows. The $H = \infty$ endpoint is MCE-IRL, so its row matches
MCE-IRL by construction.

| Horizon | Policy TV | Value RMSE | Note |
| --- | ---: | ---: | --- |
| $H = 0$ | 0.0952 | 19.53 | Max-Margin-Planning end |
| $H = 1$ | 0.0766 | 18.34 | one soft backup |
| $H = 3$ | 0.0640 | 15.10 | middle of the spectrum |
| $H = \infty$ | 0.0360 | 3.73 | matches MCE-IRL |

The comparison covers the full structural and IRL roster. See the
[high-dimensional route-choice simulation study](../../simulation_studies/highdim_route_choice.md)
for the cross-estimator table and counterfactual regret.

### Lookahead recovery

The second study draws demonstrations from a finite-lookahead planner with
a known true lookahead $h$, then fits RHIP across a sweep of estimator
horizons. The best-fitting horizon tracks the demonstrator's lookahead:
$h = 1 \to H = 1$, $h = 2 \to H = 2$, $h = 3 \to H = 3$. In every case
both endpoints are worse. See the
[horizon-recovery simulation study](../../simulation_studies/rhip_lookahead.md).
