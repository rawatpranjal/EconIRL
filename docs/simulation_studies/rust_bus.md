# Bus engine replacement

The canonical structural benchmark (Rust 1987). A single agent decides each period whether to keep paying a mileage-dependent operating cost or pay a fixed replacement cost to reset the bus engine. The data-generating process is fully known, so the table reports the exact recovered cost parameters, the distance between each estimator's policy and the true one, and the welfare lost when the recovered model is carried into three counterfactual worlds.

## The data-generating process

Mileage sits on a discrete grid $s \in \{0, \ldots, S-1\}$. Keeping the engine (action $0$) pays a per-bin operating cost and lets mileage drift up by $\Delta s \in \{0, 1, 2\}$; replacing it (action $1$) pays a flat cost and resets the engine:

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
with $\theta_{\mathrm{oc}} = 0.01$ and $\theta_{\mathrm{rc}} = 2.0$. The agent discounts at $\beta$ and faces i.i.d. logit taste shocks (scale $\sigma = 1$), so behavior solves the soft Bellman equation

$$
V(s) = \log \sum_{a} \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
\qquad \pi^*(a \mid s) \propto \exp\Bigl(u_\theta(s,a) + \beta\, \mathbb{E}\bigl[V(s') \mid s,a\bigr]\Bigr),
$$

and the panel simulates $N$ buses for $T$ periods from $\pi^*$. The figure shows the sawtooth mileage paths (rising drift, replacement resets) and the declining value of holding higher mileage. Every estimator below sees the same panels.

Harold Zurcher's bus-engine replacement problem (Rust 1987): a binary keep-or-replace choice over a discretized mileage state, with linear operating and replacement costs. `RustBusEnvironment(num_mileage_bins=20, operating_cost=0.01, replacement_cost=2.0, discount_factor=0.95)`. 500 x 80 observations, 3 replications, seed 42. True theta `[0.01, 2.0]`. Design rank 2/2, condition number 1.11e+01. Generated 2026-06-12 with econirl 0.0.4.

![Simulated trajectories and the optimal value function for Bus engine (20 mileage bins)](../_static/simulation_studies/rust_bus_dgp.png)

## Results

| Estimator | Family | Ran | Conv | Recovered params | Param RMSE | Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NFXP | structural | 3/3 | 3/3 | [0.012, 2.011] | 0.0130 | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 4.2 |
| CCP | structural | 3/3 | 3/3 | [0.011, 2.008] | 0.0125 | 0.0078 | 0.0005 | 0.0006 | 0.0004 | 0.0000 | 3.2 |
| MPEC | structural | 3/3 | 3/3 | [0.011, 2.010] | 0.0130 | 0.0074 | 0.0007 | 0.0007 | 0.0005 | 0.0000 | 11.4 |
| NNES | structural | 3/3 | 3/3 | [0.012, 2.011] | 0.0130 | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 22.6 |
| SEES | structural | 3/3 | 0/3 | [0.011, 2.010] | 0.0128 | 0.0074 | 0.0007 | 0.0007 | 0.0004 | 0.0000 | 2.9 |
| TD-CCP | structural | 3/3 | 3/3 | [0.012, 2.013] | 0.0118 | 0.0090 | 0.0010 | 0.0011 | 0.0007 | 0.0000 | 3.9 |
| UFXP | structural | 3/3 | 3/3 | [0.011, 2.009] | 0.0122 | 0.0067 | 0.0006 | 0.0006 | 0.0004 | 0.0000 | 0.2 |
| MCE-IRL | behavioral | 3/3 | 0/3 | [0.012, 2.011] | - | 0.0075 | 0.0007 | 0.0008 | 0.0005 | 0.0000 | 8.6 |
| MaxEnt-IRL | behavioral | 3/3 | 3/3 | [-0.006, 1.685] | - | 0.0649 | 0.0987 | 0.1075 | 0.0702 | 0.0003 | 9.4 |
| IQ-Learn | behavioral | 3/3 | 3/3 | [-0.016, 1.519] | - | 0.0420 | 0.4733 | 0.5252 | 0.1696 | 0.0004 | 1.9 |
| GLADIUS | behavioral | 3/3 | 3/3 | [0.029, 2.031] | - | 0.0095 | 0.0773 | 0.0795 | 0.0631 | 0.0542 | 32.7 |
| AIRL | behavioral | 3/3 | 0/3 | [0.020, 2.034] | - | 0.0528 | 0.0251 | 0.0261 | 0.0140 | 0.0025 | 132.5 |
| f-IRL | behavioral | 3/3 | 3/3 | not in theta gauge (40 values) | - | 0.0266 | 0.0536 | 0.0490 | 0.1652 | 20.4010 | 23.9 |
| Deep-MCE-IRL | behavioral | 3/3 | 3/3 | [-0.082, 0.568] | - | 0.0092 | 3.3450 | 3.2419 | 1.6305 | 0.0005 | 14.0 |
| MaxMargin-IRL | behavioral | 3/3 | 3/3 | [0.244, 0.970] | - | 0.6341 | 5.0624 | 5.0810 | 19.6480 | 10.1756 | 0.5 |
| BC | behavioral | 3/3 | 3/3 | not in theta gauge (40 values) | - | 0.0191 | 0.0045 | 0.0054 | 0.3624 | 23.8907 | 0.2 |

Param RMSE is the structural family only (recovered theta vs true, same gauge). Recovered params are shown only when the estimator's parameter vector lives in the data-generating gauge; a tabular reward or a choice-probability table is labeled rather than printed, because comparing it to theta entry by entry would be meaningless. Policy TV is total-variation distance from the true-parameter policy. Conv is the converged flag reported by the estimator itself; a conservative flag can read False while the recovered policy is accurate, so read it next to Policy TV, not alone. Regret is welfare loss (lower is better): `base` is the observed world; `A` payoff shift, `B` transition change, `C` action penalty. Estimators that recovered a reward in the linear feature gauge re-solve it under each intervention and adapt. Large Type C regret has two distinct routes: estimators with no reward in that gauge keep their frozen policy and cannot adapt, and an estimator that transfers a badly scaled reward adapts to the wrong world.

Reading the table: the structural family (NFXP, CCP, MPEC, NNES, SEES, TD-CCP) recovers the cost parameters in the same gauge as the truth, so Param RMSE applies to it alone. The IRL family is scored on behavior and regret; reward parameters from these methods live in a different gauge (reward is only partially identified from behavior), so parameter-level comparisons across the divide would be meaningless. Estimators that recover a reward in the linear feature gauge adapt under the Type A/B/C interventions; policy-only methods keep their frozen policy, which is exactly why their Type C regret is large.

## Code used

The exact construction for each estimator (configs are modest defaults with documented fixes, not tuned per cell):

### NFXP

Reference structural estimator; recovers cleanly.

```python
def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### CCP

Hotz-Miller conditional choice probabilities; recovers cleanly.

```python
def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### MPEC

Constrained MLE; recovers cleanly.

```python
def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    est = MPECEstimator(config=MPECConfig(solver="slsqp", max_iter=200, constraint_tol=1e-6),
                        compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### NNES

Neural value network plus structural MLE.

```python
def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    est = NNESEstimator(hidden_dim=64, v_epochs=800, n_outer_iterations=5,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### SEES

Solver-limited here, not model-limited: the spline value basis represents the true value function exactly, but the cost coefficients live on very different scales (a tiny per-bin operating cost against a large flat replacement cost), which stretches the optimization landscape so the default iteration limit stopped the search mid-descent. With a larger budget and an extra data-driven start it reaches the same estimates as the other structural methods.

```python
def _run_sees(env, panel):
    from econirl.estimation.sees import SEESEstimator

    # Basis must span the value function: bspline basis_dim >= num_states (20).
    # The cost coefficients live on very different scales here (per-bin
    # operating cost vs a flat replacement cost), which stretches the
    # optimization landscape: the default 500 L-BFGS iterations stop
    # mid-descent. More iterations plus a data-driven extra start reach the
    # optimum the basis already represents exactly.
    est = SEESEstimator(basis_type="bspline", basis_dim=20, warm_start_value=True,
                        penalty_weight=10.0, max_iter=3000, num_theta_starts=3,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### TD-CCP

Neural CCP with approximate value iteration.

```python
def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=15,
                                            epochs_per_avi=15, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### UFXP

Unnested fixed point (Bray; Oguz and Bray 2026) with the paper's optimal weighting (OUFXP). Bellman first-order conditions are scored with the value function eliminated before any parameter search, so the linear case is closed form; the optimal weights make it as asymptotically efficient as maximum likelihood, and standard errors come from the efficient moment variance.

```python
def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point with the paper's optimal weighting (OUFXP):
    # closed form for linear utility, MLE-efficient, with standard errors from
    # the efficient moment variance.
    est = UFXPEstimator(weights="optimal", verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### MCE-IRL

Causal maximum-entropy IRL. Its converged flag reports whether the gradient norm crossed the tolerance; the objective often plateaus first, so the flag can read False while the recovered policy is essentially exact. Read it next to Policy TV.

```python
def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### MaxEnt-IRL

Fed action-dependent features. Its gradient loop previously took a fixed scalar step, which overshoots when feature columns differ in scale by an order of magnitude (mileage cost vs a unit replacement indicator); the loop now takes adaptive per-parameter steps, the same scheme its causal cousin MCE-IRL uses. A small residual gap to MCE-IRL remains because trajectory-entropy feature matching is not the causal choice model that generated the data.

```python
def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    # Feed the action-dependent features: a state-only reward cannot represent
    # the action contrast that drives the keep/replace choice. learning_rate
    # drives the Adam step (a fixed scalar step overshoots the mileage-cost
    # coordinate, whose feature column is ~19x the replacement indicator's).
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             learning_rate=0.05, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### IQ-Learn

q_type='linear' uses the feature structure; a tabular Q-table does not propagate to unvisited states.

```python
def _run_iq_learn(env, panel):
    from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

    # q_type="linear" uses the feature structure; a tabular Q does not
    # propagate to unvisited states.
    est = IQLearnEstimator(config=IQLearnConfig(q_type="linear", divergence="chi2",
                                                alpha=3.0, max_iter=2000, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### GLADIUS

Neural Q and expected-value networks; tracks behavior.

```python
def _run_gladius(env, panel):
    from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

    est = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=300, q_hidden_dim=128,
                                                v_hidden_dim=128, q_lr=1e-4, v_lr=1e-4,
                                                patience=60, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### AIRL

reward_arg='state_action'; recovered parameters stay gauge/shaping-unidentified by design, so policy TV is the right scorecard.

```python
def _run_airl(env, panel):
    from econirl.estimation import AIRLConfig, AIRLEstimator

    # reward_arg="state_action": the default "state" marginalizes the reward
    # across actions. Recovered parameters stay gauge/shaping-unidentified by
    # design, so policy TV is the right scorecard.
    est = AIRLEstimator(config=AIRLConfig(reward_type="linear", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### f-IRL

Uses the forward-KL divergence (bounded density-ratio gradient; the chi-squared variant's unbounded gradient is unstable on near-deterministic experts) with a reward clip matched to the problem's cost scale. It recovers a tabular reward, one value per state-action pair, which tracks behavior well but lives outside the two-feature cost gauge - so it cannot be re-solved under the interventions and is scored with its frozen policy, which is why its Type C regret is large.

```python
def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    # fkl is the estimator's validated divergence for state-action cells: its
    # log density-ratio gradient is bounded, where the chi2 ratio gradient is
    # unbounded and saturates the reward clip on near-deterministic experts.
    # reward_clip=10 matches the natural cost scale (the estimator default).
    est = FIRLEstimator(f_divergence="fkl", lr=0.2, max_iter=400, reward_clip=10.0,
                        verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### Deep-MCE-IRL

Neural-reward MCE-IRL via its sklearn-style fit interface; parameters are the neural reward projected onto the linear features.

```python
def _run_deep_mce_irl(env, panel):
    from types import SimpleNamespace

    from econirl.estimators.mceirl_neural import MCEIRLNeural

    # sklearn-style .fit interface; adapted to the uniform result shape. coef_
    # is the neural reward projected onto the linear features, so the regret
    # transfer uses that projection, not the raw network.
    m = MCEIRLNeural(n_states=int(env.num_states), n_actions=int(env.num_actions),
                     discount=float(env.problem_spec.discount_factor),
                     max_epochs=200, verbose=False)
    m.fit(panel, features=np.asarray(env.feature_matrix),
          transitions=np.asarray(env.transition_matrices))
    return SimpleNamespace(parameters=m.coef_, standard_errors=None, policy=m.policy_,
                           value_function=m.value_, converged=bool(m.converged_))
```

### MaxMargin-IRL

An honest structural failure, not a tuning problem: max-margin apprenticeship learning recovers a reward direction under a unit-norm normalization with no link to the choice model's noise scale, so the policy it implies is far sharper than the truth, and the extreme asymmetry between the per-bin operating cost and the flat replacement cost makes the replacement feature dominate the margin. The resulting policy distance is structural to the method on this problem.

```python
def _run_max_margin(env, panel):
    from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator

    # Requires a reward spec (LinearReward/ActionDependentReward), not the
    # structural LinearUtility wrapper.
    est = MaxMarginIRLEstimator(max_iterations=50, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### BC

Behavioral cloning; matches observed choices but recovers no reward at all - its parameter vector is just the smoothed keep/replace frequency per mileage bin - so it cannot transfer to a counterfactual world and its Type C regret is large.

```python
def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

## Reproduce

```bash
python scripts/sim_rust_bus.py                 # run + write JSON
python scripts/sim_rust_bus.py --page          # regenerate this page
python scripts/sim_rust_bus.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/sim_rust_bus.json`. Counterfactual regret follows the package Type A (payoff shift), Type B (transition change), Type C (action penalty) taxonomy; regret = initial_distribution . (oracle_value - estimated_value), lower is better. Estimators with a recovered reward re-solve it under each intervention (transfer); estimators without one keep their fixed policy (cannot adapt).

Excluded from this run: AIRL-Het / AAIRL (designed for latent-type heterogeneity; this panel has a single agent type); MMP (dropped from the roster for cost after an exploratory fit ran orders of magnitude past its cousins' runtimes on this small problem); GAIL (did not finish a single exploratory fit within this page's per-fit budget); GCL, DeepMaxEnt-IRL, Bayesian-IRL (dropped from the page roster by scope decision to keep the comparison on the core structural and IRL families).