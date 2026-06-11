# Abstract MDP 1: sanity check

The simplest abstract problem: a small but non-trivial random MDP with an action-dependent linear reward, easy enough that a correct estimator must recover it. It is the sanity check that the whole roster works before harder regimes. Every estimator on the uniform estimate interface is run; the table reports the exact recovered parameters, recovery error, policy distance from the true policy, and counterfactual regret.

Environment: `random_mdp(num_states=8, num_actions=2, num_features=2, branching=3, discount_factor=0.9, seed=0)`. 300 x 50 observations, 3 replications. True theta `[-0.1437, 0.7872]`. Generated 2026-06-11 with econirl 0.0.4.

## Results

| Estimator | Family | Ran | Recovered params | Param RMSE | Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| NFXP | structural | 3/3 | [-0.154, 0.797] | 0.0251 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 4.0 |
| CCP | structural | 3/3 | [-0.154, 0.797] | 0.0250 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 2.0 |
| MPEC | structural | 3/3 | [-0.154, 0.797] | 0.0252 | 0.0026 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 4.8 |
| NNES | structural | 3/3 | [-0.154, 0.797] | 0.0251 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 10.9 |
| SEES | structural | 3/3 | [-0.154, 0.797] | 0.0254 | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 0.6 |
| TD-CCP | structural | 3/3 | [-0.155, 0.794] | 0.0215 | 0.0021 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 2.8 |
| MCE-IRL | behavioral | 3/3 | [-0.154, 0.797] | - | 0.0025 | 0.0002 | 0.0002 | 0.0002 | 0.0000 | 5.2 |
| MaxEnt-IRL | behavioral | 3/3 | [-0.021, 0.735] | - | 0.0095 | 0.0026 | 0.0027 | 0.0026 | 0.0000 | 14.7 |
| IQ-Learn | behavioral | 3/3 | [-0.219, 0.773] | - | 0.0370 | 0.0091 | 0.0101 | 0.0096 | 0.0000 | 1.1 |
| GLADIUS | behavioral | 3/3 | [-0.380, 0.915] | - | 0.0165 | 0.0062 | 0.0066 | 0.0044 | 0.0000 | 15.5 |
| AIRL | behavioral | 3/3 | [0.147, 0.522] | - | 0.0324 | 0.0513 | 0.0554 | 0.0297 | 0.0000 | 90.9 |
| f-IRL | behavioral | 3/3 | [-0.268, 0.222, -0.318, 0.264, -1.017, -0.187, -0.613, 0.094, -0.723, 0.322, -0.585, 0.360, -0.615, 0.485, -0.486, 0.957] | - | 0.0090 | 0.0034 | 0.0473 | 0.0729 | 61.4995 | 21.5 |
| BC | behavioral | 3/3 | [0.404, 0.596, 0.314, 0.686, 0.211, 0.789, 0.305, 0.695, 0.257, 0.743, 0.272, 0.728, 0.202, 0.798, 0.146, 0.854] | - | 0.0088 | 0.0026 | 0.0423 | 0.0676 | 61.4560 | 0.2 |

Param RMSE is the structural family only (recovered theta vs true, same gauge). Policy TV is total-variation distance from the true-parameter policy. Regret is welfare loss (lower is better): `base` is the observed world; `A` payoff shift, `B` transition change, `C` action penalty. Transfer uses the recovered reward in the linear feature gauge (theta . features): estimators that recovered such a reward re-solve it under each intervention and adapt. Estimators that return a tabular object outside that gauge (here f-IRL and behavioral cloning) are scored with their fixed policy and cannot adapt, which shows up as large Type C regret. For behavioral cloning that frozen reading is exactly correct (it recovers no reward); for a tabular-reward method it is a conservative lower bound on what the method could transfer.

## Code used

The exact construction for each estimator (configs are modest quick-run defaults, not tuned):

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

Neural value network plus structural MLE; recovers cleanly.

```python
def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    est = NNESEstimator(hidden_dim=64, v_epochs=800, n_outer_iterations=5,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### SEES

Fixed: bspline basis with basis_dim >= num_states. A fourier basis of dim 4 underfit the 8-state value function (param RMSE 0.89 -> 0.01).

```python
def _run_sees(env, panel):
    from econirl.estimation.sees import SEESEstimator

    # Basis must span the value function: bspline basis_dim >= num_states. A
    # fourier basis_dim=4 underfit the 8-state value (workflow diagnosis).
    est = SEESEstimator(basis_type="bspline", basis_dim=8, warm_start_value=True,
                        penalty_weight=10.0, compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### TD-CCP

Neural CCP with approximate value iteration; recovers cleanly.

```python
def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=15,
                                            epochs_per_avi=15, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### MCE-IRL

Causal maximum-entropy IRL; recovers behavior cleanly.

```python
def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### MaxEnt-IRL

Fixed: feed action-dependent features. A state-only reward is broadcast equally across actions and cannot represent the action contrast (policy TV 0.23 -> 0.01).

```python
def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    # Feed the action-dependent features: a state-only reward is broadcast
    # equally across actions and cannot represent the action contrast that
    # drives choice here (workflow diagnosis).
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### IQ-Learn

Fixed: q_type='linear'. A tabular Q-table does not propagate to unvisited states (policy TV 0.29 -> 0.04).

```python
def _run_iq_learn(env, panel):
    from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

    # q_type="linear" uses the feature structure; a tabular Q does not propagate
    # to unvisited states (workflow diagnosis).
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

Fixed: reward_arg='state_action'. The default 'state' marginalized the reward across actions (policy TV 0.24 -> 0.02); recovered parameters stay gauge/shaping-unidentified by design, so TV is the right scorecard.

```python
def _run_airl(env, panel):
    from econirl.estimation import AIRLConfig, AIRLEstimator

    # reward_arg="state_action": the default "state" marginalizes the reward
    # across actions and cannot represent the action contrast (workflow
    # diagnosis). AIRL accepts a reward spec, not a utility. Policy TV is fixed;
    # the recovered parameters stay gauge/shaping-unidentified by design.
    est = AIRLEstimator(config=AIRLConfig(reward_type="linear", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### f-IRL

f-divergence IRL; tracks behavior.

```python
def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    est = FIRLEstimator(f_divergence="chi2", lr=0.5, max_iter=400, reward_clip=100.0,
                        verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### BC

Behavioral cloning; matches observed choices but recovers no reward, so it cannot transfer to a counterfactual world.

```python
def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

## Reproduce

```bash
python scripts/quick_all_estimators.py --replications 3   # run + write JSON
python scripts/quick_all_estimators.py --page          # regenerate this page
python scripts/quick_all_estimators.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/quick_all_estimators.json`. Counterfactual regret follows the package Type A (payoff shift), Type B (transition change), Type C (action penalty) taxonomy; regret = initial_distribution . (oracle_value - estimated_value), lower is better. Estimators with a recovered reward re-solve it under each intervention (transfer); estimators without one keep their fixed policy (cannot adapt).

Excluded from this run: MCE-IRL-NN (uses the sklearn .fit interface, not the uniform .estimate path); GAIL (known slow (~9 min/fit); not a quick run); DeepMaxEnt-IRL (known slow (~7 min/fit); not a quick run); Bayesian-IRL (known slow (~16 min/fit); not a quick run).