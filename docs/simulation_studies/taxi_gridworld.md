# Gridworld navigation

Gridworld navigation is the home turf of the maximum-entropy IRL tradition of Ziebart's MaxEnt and its descendants, so this page weights the roster toward IRL methods. NFXP, CCP, MPEC, and UFXP run as the structural contrast. The environment also supplies a stress the bus engine does not. Every trajectory starts at the same corner and walks toward the goal, so states off that path are visited rarely or never. Methods that invert state-by-state choice frequencies feel that thinness. Methods that share strength through features or networks do not.

## The data-generating process

States are cells of an $N \times N$ grid indexed $s = \mathrm{row} \cdot N + \mathrm{col}$, with five actions (left, right, up, down, stay), deterministic moves, and an absorbing goal at the bottom-right corner. The reward has three parts: a per-step penalty, a terminal bonus when the chosen move reaches the goal, and a shaping term in the Manhattan distance $d(s)$ to the goal:

$$
u_\theta(s, a) = \theta_{\mathrm{step}}\, \mathbf{1}\{s \neq s_{\mathrm{goal}}\}
+ \theta_{\mathrm{goal}}\, \mathbf{1}\{s'(s, a) = s_{\mathrm{goal}}\}
- \theta_{\mathrm{dist}}\, \frac{d(s)}{2N},
$$

with $\theta_{\mathrm{step}} = -0.1$, $\theta_{\mathrm{goal}} = 10$, $\theta_{\mathrm{dist}} = 0.1$. The agent discounts at $\beta$ and faces i.i.d. logit taste shocks (scale $\sigma = 1$), so behavior solves the soft Bellman equation

$$
V(s) = \log \sum_{a} \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
\qquad \pi^*(a \mid s) \propto \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
$$

and every trajectory starts at the top-left corner (state 0). The figure shows the resulting paths climbing the state index toward the absorbing goal and the value function rising with proximity to it. The horizon is deliberately short (20 periods) because the goal is absorbing: once there, an agent generates no further information.

An agent starts at the top-left corner of an 8x8 grid and walks toward an absorbing goal at the bottom-right, with a per-step penalty, a terminal reward, and a distance shaping term. `GridworldEnvironment(grid_size=8, step_penalty=-0.1, terminal_reward=10.0, distance_weight=0.1, discount_factor=0.95)`. Transitions are deterministic; 64 states, 5 actions (left, right, up, down, stay). 500 x 20 observations, 3 replications, seed 7. True theta `[-0.1, 10.0, 0.1]`. Design rank 3/3, condition number 7.45e+00, action-contrast rank 1/3 (the rank that identification from choices actually uses). Generated 2026-06-12 with econirl 0.0.4.

![Simulated trajectories and the optimal value function for Gridworld 8x8](../_static/simulation_studies/taxi_gridworld_dgp.png)

## Results

| Estimator | Family | Ran | Conv | Recovered params | Param RMSE | Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxEnt-IRL | behavioral | 3/3 | 0/3 | [9.828, 9.135, 2.560] | - | 0.2567 | 30.5585 | 33.9597 | 22.1444 | 52.8451 | 22.5 |
| MCE-IRL | behavioral | 3/3 | 0/3 | [138410964683833470877696.000, -355665424254815252250624.000, -2753454791391099398127616.000] | - | 0.3632 | 35.0280 | 34.0773 | 35.2199 | 26.0297 | 251.1 |
| Deep-MCE-IRL | behavioral | 3/3 | 3/3 | [0.358, 0.415, 0.375] | - | 0.3986 | 70.4902 | 84.9226 | 38.4590 | 50.0921 | 11.1 |
| AIRL | behavioral | 3/3 | 0/3 | [-1.830, -1.938, 0.178] | - | 0.6152 | 93.9839 | 92.0447 | 41.3479 | 49.9806 | 113.7 |
| IQ-Learn | behavioral | 3/3 | 3/3 | [0.251, 2.996, -18.648] | - | 0.6165 | 100.8896 | 98.0318 | 68.2383 | 53.7943 | 2.1 |
| f-IRL | behavioral | 3/3 | 3/3 | different parameterization (320 values) | - | 0.0509 | 0.0505 | 0.1265 | 58.6750 | 22.6898 | 28.4 |
| GLADIUS | behavioral | 3/3 | 3/3 | [0.000, 2.367, 0.000] | - | 0.2208 | 5.6228 | 6.7224 | 21.6122 | 49.9853 | 25.6 |
| BC | behavioral | 3/3 | 3/3 | different parameterization (320 values) | - | 0.1298 | 2.1161 | 2.1524 | 48.9540 | 24.3452 | 0.2 |
| NFXP | structural | 3/3 | 3/3 | [-15615.134, -3419.303, -9623.201] | 10826.8304 | 0.0004 | 0.0029 | 0.0033 | 15.1643 | 0.0512 | 12.0 |
| CCP | structural | 3/3 | 3/3 | [-88.773, 87.366, -136.175] | 106.1060 | 0.0081 | 0.0028 | 0.0032 | 14.1712 | 0.0495 | 3.1 |
| MPEC | structural | 3/3 | 3/3 | [-108.727, 30.044, -103.928] | 89.8227 | 0.0004 | 0.0029 | 0.0033 | 11.0472 | 0.0512 | 0.9 |
| UFXP | structural | 3/3 | 0/3 | [-2.333, 2.456, 3.726] | 6.1309 | 0.0388 | 30.4731 | 30.5956 | 23.6958 | 27.0447 | 0.1 |

Param RMSE is reported for the structural family only. Those estimators share the parameterization of the true model, so the comparison is meaningful. Recovered params are printed only in that same parameterization. A tabular reward or a choice-probability table is labeled instead of printed. Policy TV is the total-variation distance from the true-parameter policy. Conv is the estimator's own convergence flag. A conservative flag can read False while the policy is accurate, so read it next to Policy TV. Regret is welfare loss, lower is better. Base is the observed world. Type A shifts a payoff, Type B changes the transitions, Type C penalizes an action. Structural estimators re-solve the model and adapt. Behavioral estimators keep their old policy.

The headline is the gap between Policy TV and Param RMSE for the structural family. Behavior is near-perfect while the parameter estimates are orders of magnitude from the truth. That is not estimation error. It is non-identification. The raw feature design has full rank, but the action-contrast design has rank 1. The step-penalty and distance features take the same value for every action at a state, so they difference out of every choice probability. Choice data can only identify the contrast design. Any parameter vector on that two-dimensional ridge produces the same behavior, so each estimator reports an arbitrary ridge point. The practical lesson is to check the rank of the action-differenced features before estimating, not just the raw design.

One caveat on regret for this page. Transitions are deterministic, so the Type B intervention is a stark change rather than a perturbation. Read Type B as a stress test of reward transfer under completely new dynamics, not a local robustness check.

## Notes per estimator

**MaxEnt-IRL.** The Ziebart tradition this environment comes from. Matches discounted feature counts; with two of three features state-only, most of its objective is insensitive to the choice contrasts the data carry.

**MCE-IRL.** Causal maximum-entropy IRL. Two of its three reward directions are unidentified here, because the features are state-only. Its gradient ascent can drift far along them. In one replication of three the policy collapsed outright. Read the per-rep records, not just the mean.

**Deep-MCE-IRL.** Neural-reward MCE-IRL via its sklearn-style fit interface. Parameters are the neural reward projected onto the linear features.

**AIRL.** Uses reward_arg='state_action'. The recovered reward is in its own parameterization by design, so policy TV is the right scorecard. Even behavior is hard here. The discriminator sees mostly corridor states.

**IQ-Learn.** q_type='linear' ties its Q function to the same feature basis, so it inherits the contrast-rank problem on top of thin coverage.

**f-IRL.** Recovers a tabular reward, one value per state-action pair. That does not depend on the deficient feature basis at all, and it posts the strongest behavioral score on this page.

**GLADIUS.** Neural Q and expected-value networks. Tracks behavior where data exists.

**BC.** Behavioral cloning. It matches observed choices where data exists, falls back to uniform where it does not, and recovers no reward.

**NFXP.** The structural contrast, exact MLE. It reproduces the true policy almost perfectly while reporting parameters far from the truth. The likelihood is flat along the two state-only feature directions, so the parameter numbers are arbitrary points on a ridge, not estimation error.

**CCP.** Structural contrast. Same flat likelihood as NFXP, plus inverted choice probabilities estimated from a concentrated state distribution.

**MPEC.** Structural contrast. Constrained MLE on the same ridge.

**UFXP.** Unnested fixed point (Bray; Oguz and Bray 2026) with optimal weighting (OUFXP). Its moment system is built from the action-contrast features, so the rank-1 design leaves two directions to the minimum-norm solution. Behavior stays close. Parameters are pinned only in the identified direction.

## Reproduce

```bash
python scripts/sim_taxi_gridworld.py                 # run + write JSON
python scripts/sim_taxi_gridworld.py --page          # regenerate this page
python scripts/sim_taxi_gridworld.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/sim_taxi_gridworld.json`. Counterfactual regret follows the package Type A (payoff shift), Type B (transition change), Type C (action penalty) taxonomy; regret = initial_distribution . (oracle_value - estimated_value), lower is better. Estimators with a recovered reward re-solve it under each intervention (transfer); estimators without one keep their fixed policy (cannot adapt).

Excluded from this run: SEES (its spline value basis is built for an ordered 1-D state index. A 2-D grid breaks that geometry, so running it here would be misspecification by construction); NNES, TD-CCP (the structural contrast is carried by NFXP/CCP/MPEC/UFXP here. The full structural roster runs on the bus engine and abstract MDP pages); MMP, GAIL, GCL, DeepMaxEnt-IRL, Bayesian-IRL (dropped from the study's rosters by scope decision (MMP and GAIL also failed a 20-30 minute single-fit budget on the bus engine page)).