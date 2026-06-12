# Abstract MDP 1: sanity check

The simplest abstract problem. A small but non-trivial random MDP with an action-dependent linear reward, easy enough that a correct estimator must recover it. It is the sanity check that the whole roster works before the harder regimes. Every estimator on the uniform estimate interface runs here. The table reports the exact recovered parameters, the recovery error, the policy distance from the truth, and the counterfactual regret.

Environment: `random_mdp(num_states=8, num_actions=2, num_features=2, branching=3, discount_factor=0.9, seed=0)`. 300 x 50 observations, 3 replications. True theta `[-0.1437, 0.7872]`. Generated 2026-06-12 with econirl 0.0.4.

## The data-generating process

One Garnet-style MDP is drawn from the seed and held fixed. Each state-action pair reaches a uniform random subset of $b$ states with Dirichlet weights, mixed with a small self-loop mass $\ell$:

$$
P(s' \mid s, a) \;=\; (1-\ell)\, D_{s,a}(s') \;+\; \ell\, \mathbf{1}\{s'=s\},
\qquad D_{s,a} \sim \mathrm{Dirichlet}(\mathbf{1}_b),\quad b = 3,\ \ell = 0.05 .
$$

The reward is linear in features of the normalized state index $x_s = s/(S-1)$. Action $0$ is a zeroed outside option, the identification anchor. For action $1$,

$$
u_\theta(s,a) = \theta^\top \varphi(s,a),
\qquad \varphi(s,1) = \bigl(1,\ x_s + 1\bigr),
\qquad \theta \sim \mathcal{N}(0,\ 0.25\, I_2).
$$

The agent discounts at $\beta = 0.9$ and faces i.i.d. logit taste shocks (scale $\sigma = 1$), so behavior solves the soft Bellman equation

$$
V(s) = \log \sum_{a} \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
\qquad \pi^*(a \mid s) \propto \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
$$

and the data are $N$ independent agents simulated for $T$ periods from $\pi^*$ and the transition law. The figure shows what that produces. State paths mix across the whole space, and the optimal value function varies smoothly in the state index.

![Simulated trajectories and the optimal value function](../_static/simulation_studies/abstract_mdp_1_dgp.png)

## Results

| Estimator | Family | Ran | Recovered params | Param RMSE | Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| NFXP | structural | 3/3 | [-0.154, 0.797] | 0.0251 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 4.4 |
| CCP | structural | 3/3 | [-0.154, 0.797] | 0.0250 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 2.2 |
| MPEC | structural | 3/3 | [-0.154, 0.797] | 0.0252 | 0.0026 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 5.4 |
| NNES | structural | 3/3 | [-0.154, 0.797] | 0.0251 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 12.6 |
| SEES | structural | 3/3 | [-0.154, 0.797] | 0.0254 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 0.8 |
| TD-CCP | structural | 3/3 | [-0.155, 0.794] | 0.0215 | 0.0021 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 3.5 |
| UFXP | structural | 3/3 | [-0.155, 0.796] | 0.0248 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 0.1 |
| MCE-IRL | behavioral | 3/3 | [-0.154, 0.797] | - | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 5.8 |
| MaxEnt-IRL | behavioral | 3/3 | [-0.348, 0.923] | - | 0.0101 | 0.0056 | 0.0062 | 0.0016 | 0.0000 | 14.4 |
| IQ-Learn | behavioral | 3/3 | [-0.219, 0.773] | - | 0.0370 | 0.0091 | 0.0101 | 0.0096 | 0.0000 | 1.4 |
| GLADIUS | behavioral | 3/3 | [-0.380, 0.915] | - | 0.0165 | 0.0062 | 0.0066 | 0.0044 | 0.0000 | 17.1 |
| AIRL | behavioral | 3/3 | [0.147, 0.522] | - | 0.0324 | 0.0513 | 0.0554 | 0.0297 | 0.0000 | 98.4 |
| f-IRL | behavioral | 3/3 | different parameterization (16 values) | - | 0.0091 | 0.0034 | 0.0462 | 0.0736 | 61.4508 | 23.0 |
| BC | behavioral | 3/3 | different parameterization (16 values) | - | 0.0088 | 0.0026 | 0.0423 | 0.0676 | 61.4560 | 0.2 |

Param RMSE is reported for the structural family only. Those estimators share the parameterization of the true model. Policy TV is the total-variation distance from the true-parameter policy. Regret is welfare loss, lower is better. Base is the observed world. Type A shifts a payoff, Type B changes the transitions, Type C penalizes an action. Estimators that recovered a linear feature reward re-solve it under each intervention and adapt. f-IRL and behavioral cloning return tabular objects instead, so they are scored with their fixed policy. That shows up as large Type C regret. For behavioral cloning the fixed reading is exact, because it recovers no reward. For a tabular-reward method it is a conservative reading.

## Notes per estimator

**NFXP.** Reference structural estimator; recovers cleanly.

**CCP.** Hotz-Miller conditional choice probabilities; recovers cleanly.

**MPEC.** Constrained MLE; recovers cleanly.

**NNES.** Neural value network plus structural MLE; recovers cleanly.

**SEES.** Fixed: bspline basis with basis_dim >= num_states. A fourier basis of dim 4 underfit the 8-state value function (param RMSE 0.89 -> 0.025).

**TD-CCP.** Neural CCP with approximate value iteration; recovers cleanly.

**UFXP.** Unnested fixed point (Bray; Oguz and Bray 2026) with the paper's optimal weighting (OUFXP): closed form for linear utility, as asymptotically efficient as maximum likelihood, standard errors from the efficient moment variance.

**MCE-IRL.** Causal maximum-entropy IRL; recovers behavior cleanly.

**MaxEnt-IRL.** Fixed: feed action-dependent features. A state-only reward is broadcast equally across actions and cannot represent the action contrast (policy TV 0.23 -> 0.01).

**IQ-Learn.** Fixed: q_type='linear'. A tabular Q-table does not propagate to unvisited states (policy TV 0.29 -> 0.04).

**GLADIUS.** Neural Q and expected-value networks; tracks behavior.

**AIRL.** Fixed: reward_arg='state_action'. The default 'state' marginalized the reward across actions (policy TV 0.24 -> 0.02); recovered parameters are identified only up to shaping transformations by design, so TV is the right scorecard.

**f-IRL.** f-divergence IRL; tracks behavior.

**BC.** Behavioral cloning; matches observed choices but recovers no reward, so it cannot transfer to a counterfactual world.

Configs are modest quick-run defaults, not tuned.

## Reproduce

```bash
python scripts/quick_all_estimators.py --replications 3   # run + write JSON
python scripts/quick_all_estimators.py --page          # regenerate this page
python scripts/quick_all_estimators.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/quick_all_estimators.json`. Counterfactual regret follows the package Type A (payoff shift), Type B (transition change), Type C (action penalty) taxonomy; regret = initial_distribution . (oracle_value - estimated_value), lower is better. Estimators with a recovered reward re-solve it under each intervention (transfer); estimators without one keep their fixed policy (cannot adapt).

Excluded from this run: MCE-IRL-NN (uses the sklearn .fit interface, not the uniform .estimate path); GAIL (known slow (~9 min/fit); not a quick run); DeepMaxEnt-IRL (known slow (~7 min/fit); not a quick run); Bayesian-IRL (known slow (~16 min/fit); not a quick run).