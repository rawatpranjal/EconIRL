# Bus engine replacement

The canonical structural benchmark (Rust 1987). A single agent decides each period whether to keep paying a mileage-dependent operating cost or pay a fixed replacement cost to reset the bus engine. The data-generating process is fully known, so the table reports the exact recovered cost parameters, the distance between each estimator's policy and the true one, and the welfare lost when the recovered model is carried into three counterfactual worlds.

## The data-generating process

Mileage sits on a discrete grid $s \in \{0, \ldots, S-1\}$. Keeping the engine (action $0$) pays a per-bin operating cost and lets mileage drift up by $\Delta s \in \{0, 1, 2\}$. Replacing it (action $1$) pays a flat cost and resets the engine:

$$
u_\theta(s, a) =
\begin{cases}
-\theta_{\mathrm{oc}}\, s & a = 0 \ (\text{keep}) \\
-\theta_{\mathrm{rc}} & a = 1 \ (\text{replace}),
\end{cases}
\qquad
P(s' \mid s, 1) = p_{\Delta s'},\ s' \in \{0, 1, 2\},
$$

where replacement resets the engine and the same one-period drift $p = (p_0, p_1, p_2)$ then applies from zero, so the post-replacement state lands on $\{0, 1, 2\}$ rather than exactly on zero.

The true parameters are $\theta_{\mathrm{oc}} = 0.01$ and $\theta_{\mathrm{rc}} = 2.0$. The agent discounts at $\beta$ and faces i.i.d. logit taste shocks (scale $\sigma = 1$), so behavior solves the soft Bellman equation

$$
V(s) = \log \sum_{a} \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
\qquad \pi^*(a \mid s) \propto \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
$$

and the panel simulates $N$ buses for $T$ periods from $\pi^*$. The figure shows the sawtooth mileage paths (rising drift, replacement resets) and the declining value of holding higher mileage. Every estimator below sees the same panels.

Harold Zurcher's bus-engine replacement problem (Rust 1987): a binary keep-or-replace choice over a discretized mileage state, with linear operating and replacement costs. `RustBusEnvironment(num_mileage_bins=20, operating_cost=0.01, replacement_cost=2.0, discount_factor=0.95)`. 500 x 80 observations, 3 replications, seed 42. True theta `[0.01, 2.0]`. Design rank 2/2, condition number 1.11e+01, action-contrast rank 2/2 (the rank that identification from choices actually uses). Generated 2026-06-12 with econirl 0.0.4.

![Simulated trajectories and the optimal value function for Bus engine (20 mileage bins)](../_static/simulation_studies/rust_bus_dgp.png)

## Estimators and data

| Estimator | Family | Uses transitions $P(s'\mid s,a)$ | Transferable reward | Standard errors |
|---|---|---|---|---|
| NFXP | structural | yes | yes | yes |
| CCP | structural | yes | yes | yes |
| MPEC | structural | yes | yes | yes |
| NNES | structural | yes | yes | no |
| SEES | structural | yes | yes | no |
| TD-CCP | structural | yes | yes | no |
| UFXP | structural | yes | yes | yes |
| MCE-IRL | behavioral | yes | yes | no |
| MaxEnt-IRL | behavioral | yes | yes | no |
| IQ-Learn | behavioral | yes | yes | no |
| GLADIUS | behavioral | yes | yes | no |
| AIRL | behavioral | yes | yes | no |
| Deep-MCE-IRL | behavioral | yes | yes | no |

Uses transitions is whether the estimator reads the transition kernel; model-free learners do not. Transferable reward is whether it recovers a reward that re-solves under a counterfactual. Standard errors is whether it returns inference. The last two are read from the run.

## Results

| Estimator | Family | Ran | Conv | Recovered params | Param RMSE | Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NFXP | structural | 3/3 | 3/3 | [0.012, 2.011] | 0.0130 | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 4.2 |
| CCP | structural | 3/3 | 3/3 | [0.011, 2.008] | 0.0125 | 0.0078 | 0.0005 | 0.0006 | 0.0004 | 0.0000 | 3.2 |
| MPEC | structural | 3/3 | 3/3 | [0.012, 2.011] | 0.0130 | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 0.8 |
| NNES | structural | 3/3 | 3/3 | [0.012, 2.011] | 0.0130 | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 22.6 |
| SEES | structural | 3/3 | 0/3 | [0.011, 2.010] | 0.0128 | 0.0074 | 0.0007 | 0.0007 | 0.0004 | 0.0000 | 2.9 |
| TD-CCP | structural | 3/3 | 3/3 | [0.012, 2.013] | 0.0118 | 0.0090 | 0.0010 | 0.0011 | 0.0007 | 0.0000 | 3.9 |
| UFXP | structural | 3/3 | 3/3 | [0.011, 2.009] | 0.0122 | 0.0067 | 0.0006 | 0.0006 | 0.0004 | 0.0000 | 0.2 |
| MCE-IRL | behavioral | 3/3 | 0/3 | [0.012, 2.011] | - | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 8.6 |
| MaxEnt-IRL | behavioral | 3/3 | 3/3 | [-0.006, 1.685] | - | 0.0649 | 0.0987 | 0.1075 | 0.0702 | 0.0003 | 9.4 |
| IQ-Learn | behavioral | 3/3 | 3/3 | [-0.016, 1.519] | - | 0.0420 | 0.4733 | 0.5252 | 0.1696 | 0.0004 | 1.9 |
| GLADIUS | behavioral | 3/3 | 3/3 | [0.029, 2.031] | - | 0.0095 | 0.0773 | 0.0795 | 0.0631 | 0.0542 | 32.7 |
| AIRL | behavioral | 3/3 | 0/3 | [0.020, 2.034] | - | 0.0528 | 0.0251 | 0.0261 | 0.0140 | 0.0025 | 132.5 |
| Deep-MCE-IRL | behavioral | 3/3 | 3/3 | [-0.082, 0.568] | - | 0.0092 | 3.3450 | 3.2419 | 1.6305 | 0.0005 | 14.0 |

Param RMSE covers the structural family only, which shares the parameterization of the true model. Policy TV is the distance between estimated and true choice probabilities, lower is better. Conv is the estimator's own convergence indicator. A cautious estimator can report False while the recovered policy is accurate. Regret base is welfare lost in the observed environment. Types A, B, and C are welfare lost after a change. Type A shifts a payoff, Type B changes the transitions, Type C penalizes an action. Estimators with a recovered reward re-solve it and adapt. Those without one keep their old policy.

![Policy total variation per estimator for Bus engine (20 mileage bins)](../_static/simulation_studies/rust_bus_results.png)

The structural family (NFXP, CCP, MPEC, NNES, SEES, TD-CCP, UFXP) recovers the cost parameters on the same scale as the truth, so Param RMSE applies to it alone. MCE-IRL uses the same linear cost features and recovers the same scale, but the IRL family is scored on behavior and regret because reward is only partially identified from behavior in general. Estimators that recover a transferable reward adapt under the interventions. Policy-only methods keep their old policy, which is why their Type C regret is large.

## Scaling

The same study at three mileage-grid sizes (15, 20, 30 bins). Each line is one estimator: fit time on the left, policy total variation on the right. The structural methods stay cheap and accurate across sizes. The neural and IRL methods cost more and are less accurate. These are small fits, so the compute lines reflect fixed overhead as much as problem size, and the trend is not a clean monotone curve.

![Fit time and policy total variation against the number of states](../_static/simulation_studies/rust_bus_scaling.png)

## Reward and structure

Mileage is the natural ordering, so reward plots cleanly against the state index. The keep cost slopes down with mileage. The replacement cost is flat. Where they cross is near the replacement threshold. The recovered reward (dashed) tracks the true reward, and the optimal value falls as mileage rises.

![Reward against mileage, keep versus replace, with optimal value](../_static/simulation_studies/rust_bus_reward_curve.png)

The same reward as a state-by-action heatmap puts the true and recovered rewards side by side on one color scale.

![True and recovered reward heatmaps](../_static/simulation_studies/rust_bus_reward.png)

## Notes per estimator

**UFXP.** Unnested fixed point (Bray; Oguz and Bray 2026) with the paper's optimal weighting. The value function is eliminated before any parameter search, so the linear case is closed form and as efficient as maximum likelihood.

**MCE-IRL.** Its convergence indicator reports whether the gradient norm crossed the tolerance. The objective often plateaus first, so it can read False while the policy is essentially exact.

**MaxEnt-IRL.** It trails MCE-IRL because trajectory-entropy matching is not the causal choice model that generated the data.

**IQ-Learn.** Uses the linear feature structure. A tabular Q-table would not propagate to unvisited states.

## Reproduce

```bash
python scripts/sim_rust_bus.py                 # run + write JSON
python scripts/sim_rust_bus.py --page          # regenerate this page
python scripts/sim_rust_bus.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/sim_rust_bus.json`.

Not shown on this page: AIRL-Het / AAIRL (designed for latent-type heterogeneity; this panel has one agent type); MMP, GAIL (too slow for this page's per-fit budget); GCL, DeepMaxEnt-IRL, Bayesian-IRL (research code, not benchmarked); MaxMargin-IRL (its unit-norm reward direction has no link to the choice model's noise scale, so it is not a like-for-like baseline on this problem; it ran 3/3 and its raw records remain in the results file); f-IRL, BC (they recover a tabular reward and a choice-probability table, objects in a different parameterization; their raw records remain in the results file).