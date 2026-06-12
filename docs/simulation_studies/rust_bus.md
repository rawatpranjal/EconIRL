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
| MaxMargin-IRL | behavioral | 3/3 | 3/3 | [0.244, 0.970] | - | 0.6341 | 5.0624 | 5.0810 | 19.6480 | 10.1756 | 0.5 |

Param RMSE is reported for the structural family only. Those estimators share the parameterization of the true model, so the comparison is meaningful. Recovered params are printed only in that same parameterization. A tabular reward or a choice-probability table is labeled instead of printed. Policy TV is the total-variation distance from the true-parameter policy. Conv is the estimator's own convergence flag. A conservative flag can read False while the policy is accurate, so read it next to Policy TV. Regret is welfare loss, lower is better. Base is the observed world. Type A shifts a payoff, Type B changes the transitions, Type C penalizes an action. Structural estimators re-solve the model and adapt. Behavioral estimators keep their old policy.

The structural family (NFXP, CCP, MPEC, NNES, SEES, TD-CCP, UFXP) recovers the cost parameters on the same scale as the truth, so Param RMSE applies to it alone. The IRL family is scored on behavior and regret. Its reward parameters are in a different parameterization, because reward is only partially identified from behavior. Estimators that recover a transferable reward adapt under the interventions. Policy-only methods keep their old policy, which is why their Type C regret is large.

## Notes per estimator

**NFXP.** Reference structural estimator; recovers cleanly.

**CCP.** Hotz-Miller conditional choice probabilities; recovers cleanly.

**MPEC.** Constrained MLE; recovers cleanly.

**NNES.** Neural value network plus structural MLE.

**SEES.** Solver-limited here, not model-limited. The spline basis represents the true value function exactly, but the two cost coefficients live on very different scales. That stretches the optimization landscape, and the default iteration limit stopped the search mid-descent. With a larger budget and an extra data-driven start it matches the other structural methods.

**TD-CCP.** Neural CCP with approximate value iteration.

**UFXP.** Unnested fixed point (Bray; Oguz and Bray 2026) with the paper's optimal weighting (OUFXP). The value function is eliminated before any parameter search, so the linear case is closed form. The optimal weights make it as efficient as maximum likelihood, and the standard errors come from the same theory.

**MCE-IRL.** Causal maximum-entropy IRL. Its converged flag reports whether the gradient norm crossed the tolerance. The objective often plateaus first, so the flag can read False while the recovered policy is essentially exact. Read it next to Policy TV.

**MaxEnt-IRL.** Fed action-dependent features. Its gradient loop previously took a fixed scalar step, which overshoots when feature columns differ in scale by an order of magnitude. The loop now takes adaptive per-parameter steps, the same scheme MCE-IRL uses. A small residual gap to MCE-IRL remains because trajectory-entropy matching is not the causal choice model that generated the data.

**IQ-Learn.** q_type='linear' uses the feature structure. A tabular Q-table does not propagate to unvisited states.

**GLADIUS.** Neural Q and expected-value networks; tracks behavior.

**AIRL.** Uses reward_arg='state_action'. The recovered reward is in its own parameterization by design, so policy TV is the right scorecard.

**Deep-MCE-IRL.** Neural-reward MCE-IRL via its sklearn-style fit interface; parameters are the neural reward projected onto the linear features.

**MaxMargin-IRL.** A structural failure, not a tuning problem. Max-margin apprenticeship learning recovers a reward direction under a unit-norm constraint, with no link to the choice model's noise scale. The policy it implies is far sharper than the truth. The flat replacement cost also dominates the margin against the small per-bin operating cost. The policy distance is inherent to the method on this problem.

## Reproduce

```bash
python scripts/sim_rust_bus.py                 # run + write JSON
python scripts/sim_rust_bus.py --page          # regenerate this page
python scripts/sim_rust_bus.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/sim_rust_bus.json`. Counterfactual regret follows the package Type A (payoff shift), Type B (transition change), Type C (action penalty) taxonomy; regret = initial_distribution . (oracle_value - estimated_value), lower is better. Estimators with a recovered reward re-solve it under each intervention (transfer); estimators without one keep their fixed policy (cannot adapt).

Excluded from this run: AIRL-Het / AAIRL (designed for latent-type heterogeneity; this panel has a single agent type); MMP (dropped from the roster for cost after an exploratory fit ran orders of magnitude past its cousins' runtimes on this small problem); GAIL (did not finish a single exploratory fit within this page's per-fit budget); GCL, DeepMaxEnt-IRL, Bayesian-IRL (dropped from the page roster by scope decision to keep the comparison on the core structural and IRL families); f-IRL, BC (dropped from this page's display by scope decision. Both recover objects in a different parameterization, a tabular reward and a choice-probability table. Their rows would invite meaningless parameter comparisons here. Their raw records remain in the results file).