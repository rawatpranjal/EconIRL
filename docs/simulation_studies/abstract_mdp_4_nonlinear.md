# Abstract MDP 4

The reward has an interaction effect. The true utility multiplies two features. The estimators receive the two features but never their product, so a linear utility is misspecified by construction. The omitted term is deliberately strong here, larger than the main effects, so the cost is visible. The question is what that cost is. The table reports the distance from the true choices and the counterfactual regret.

Environment: a 24-state, 3-action MDP with sparse random transitions, drawn once at seed 404. 300 x 50 observations; the 3 replications resample the panel from that one environment. Generated 2026-06-12 with econirl 0.0.6.

## The data-generating process

Each state-action pair reaches a random subset of $b$ states with Dirichlet weights:

$$
P(s' \mid s, a) = D_{s,a}(s'), \qquad D_{s,a} \sim \mathrm{Dirichlet}(\mathbf{1}_b), \quad b = 4.
$$

Two features vary smoothly in the normalized state index $x_s = s/(S-1)$. Action $0$ is a zeroed outside option, the identification anchor. For the other actions the features are

$$
\varphi(s,1) = \bigl(x_s,\ \sin \pi x_s\bigr), \qquad \varphi(s,2) = \bigl(1-x_s,\ \cos \pi x_s\bigr).
$$

The true reward adds the product of the two features, the interaction the estimators do not model:

$$
u(s,a) = \theta_0\, \varphi_0(s,a) + \theta_1\, \varphi_1(s,a) + \gamma\, \varphi_0(s,a)\, \varphi_1(s,a), \qquad \theta = (1.0, -0.8),\ \gamma = 2.5.
$$

A linear utility fits $\theta_0 \varphi_0 + \theta_1 \varphi_1$ and has no term for the product. The neural-reward methods learn a reward or value network over the same two features and can form it. The interaction weight is set above the main effects on purpose, to make the misspecification show. A weaker interaction shrinks the gap, and at $\gamma = 0$ the linear utility is correct and recovers the reward. The agent discounts at $\beta = 0.95$ and faces logit taste shocks, so behavior solves the soft Bellman equation. The figure shows the simulated paths and the optimal value function.

![Simulated trajectories and the optimal value function](../_static/simulation_studies/abstract_mdp_4_dgp.png)

## Results

| Estimator | Reward | Ran | Conv | Policy TV | Transfer | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| NFXP | linear | 3/3 | 3/3 | 0.1049 | yes | 0.9687 | 0.9670 | 0.9012 | 0.6432 | 2.8 |
| CCP | linear | 3/3 | 3/3 | 0.1046 | yes | 0.9658 | 0.9630 | 0.9025 | 0.6299 | 2.5 |
| MPEC | linear | 3/3 | 3/3 | 0.1049 | yes | 0.9687 | 0.9670 | 0.9012 | 0.6432 | 0.4 |
| NNES | linear | 3/3 | 3/3 | 0.1049 | yes | 0.9687 | 0.9670 | 0.9012 | 0.6432 | 13.5 |
| SEES | linear | 3/3 | 1/3 | 0.1042 | yes | 0.9698 | 0.9678 | 0.8981 | 0.6391 | 1.2 |
| TD-CCP | linear | 3/3 | 3/3 | 0.1094 | yes | 1.0078 | 1.0051 | 0.8672 | 0.6143 | 4.5 |
| UFXP | linear | 3/3 | 3/3 | 0.1085 | yes | 0.9961 | 0.9945 | 0.8760 | 0.6289 | 0.3 |
| MCE-IRL | linear | 3/3 | 0/3 | 0.1049 | yes | 0.9687 | 0.9670 | 0.9012 | 0.6432 | 8.5 |
| MaxEnt-IRL | linear | 3/3 | 3/3 | 0.1046 | yes | 0.9672 | 0.9656 | 0.9064 | 0.6480 | 8.8 |
| IQ-Learn | linear | 3/3 | 3/3 | 0.1281 | yes | 1.1260 | 1.1290 | 0.8963 | 0.6671 | 1.5 |
| f-IRL | tabular | 3/3 | 3/3 | 0.0195 | no | 0.0291 | 0.0649 | 0.3983 | 71.1407 | 23.1 |
| BC | none | 3/3 | 3/3 | 0.0182 | no | 0.0246 | 0.0579 | 0.3862 | 70.8517 | 0.1 |
| GLADIUS | neural | 3/3 | 3/3 | 0.0379 | yes | 1.1613 | 1.1506 | 0.8980 | 0.5868 | 12.8 |
| AIRL | neural | 3/3 | 0/3 | 0.0218 | no | 0.0351 | 0.0735 | 0.4341 | 71.3552 | 114.3 |
| Deep MCE-IRL | neural | 3/3 | 3/3 | 0.0217 | no | 0.0391 | 0.0717 | 0.4152 | 70.9284 | 26.0 |
| Neural UFXP | neural | 3/3 | 3/3 | 0.0204 | no | 0.0281 | 0.0566 | 0.3667 | 69.8855 | 1.1 |

The interaction costs two ways. The structural estimators land together: NFXP, CCP, MPEC, NNES, SEES, TD-CCP, and UFXP all sit near a policy distance of 0.10, the residual a linear utility leaves, and their re-solved reward loses close to one unit of welfare. The maximum-entropy IRL methods sit there too. The methods with a richer reward or policy class learn the product and match the choices to about 0.02: the neural-reward Deep MCE-IRL and AIRL, f-IRL with a free tabular reward, and Neural UFXP, which trains a network utility through the same unnested fixed point the linear UFXP uses. The benchmark re-solves only linear-in-feature rewards, so under the interventions these methods are scored on their fixed policy, not on a re-solve of what they learned. GLADIUS matches the choices but projects its reward back onto the linear features, so even its baseline regret is as large as the linear family's. BC clones the choices and estimates no reward at all.

Reward marks what the method fits: a linear utility, a reward or value network, a free tabular reward (one value per state-action pair), or no reward at all (a cloned policy). Policy TV is the distance between estimated and true choice probabilities, lower is better. The value level is omitted: the reward is identified only up to transformations that leave behavior unchanged, so a value error across families would not compare like with like. Conv is the estimator's own convergence indicator; it does not track recovery here. A cautious estimator can report False while the policy is accurate, which is exactly the AIRL case below.

Regret base is welfare lost in the observed environment. Types A, B, and C are welfare lost after a change: Type A shifts a payoff, Type B changes the transitions, Type C penalizes an action. Transfer says whether the method re-solved a recovered reward (yes) or held a fixed policy (no). The benchmark re-solves only linear-in-feature rewards, so a method that learns a neural or tabular reward shows no here even though its reward could transfer in principle; this is a limit of the test, not of the method. The two modes are not comparable on Types A, B, and C: a fixed policy cannot adapt to any change, so it pays the same large Type C the oracle's own fixed policy pays (about 71). That figure marks no re-solve, not a worse estimate. Read the counterfactual columns within a transfer mode, not across.

## Notes per estimator

**UFXP.** The linear special case. It cannot form the product, so it sits with the linear family. The paper that introduces UFXP, Oguz and Bray (2026), trains a neural utility through the same unnested fixed point; that is the Neural UFXP row below.

**f-IRL.** Learns a free tabular reward, one value per state-action pair, not a linear utility. That is why it tracks the choices on a nonlinear reward. The benchmark re-solves only linear-in-feature rewards, so this tabular reward is not transferred and its counterfactual stays on the fixed policy.

**BC.** Clones the observed choice frequencies. It matches behavior with no reward, so it has nothing to carry to a counterfactual.

**GLADIUS.** Learns the behavior through a value network (policy TV 0.04), then projects the reward back onto the linear features. Its regret is scored on that projected linear reward, not on the neural policy the policy TV measures, and the projection cannot hold the interaction, so even its baseline regret is as large as the linear family's.

**Neural UFXP.** The same unnested fixed point as UFXP, but the utility is a network trained on the projected first-order conditions, with no Bellman solve in the loop. It learns the interaction and matches the choices where the linear UFXP cannot.

## Reproduce

```bash
python scripts/sim_abstract_mdp_4.py --replications 3
python scripts/sim_abstract_mdp_4.py --page
python scripts/sim_abstract_mdp_4.py --verify
```

Raw facts: `validation/results/sim_abstract_mdp_4.json`.

Excluded from this run: GAIL (known slow (~9 min/fit); not run here); DeepMaxEnt-IRL (known slow (~7 min/fit); not run here); Bayesian-IRL (known slow (~16 min/fit); not run here).