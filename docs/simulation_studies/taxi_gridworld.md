# Gridworld navigation

Gridworld navigation is the home turf of the maximum-entropy IRL tradition (Ziebart's MaxEnt and its descendants), so this page weights the roster toward IRL methods, with NFXP, CCP, MPEC, and UFXP as the structural contrast. The environment also supplies a stress the bus engine does not: every trajectory starts at the same corner and walks toward the goal, so states far from the start-to-goal path are visited rarely or never. Methods that rely on inverting state-by-state choice frequencies feel that thinness; methods that share strength through features or networks do not.

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
| f-IRL | behavioral | 3/3 | 3/3 | not in theta gauge (320 values) | - | 0.0509 | 0.0505 | 0.1265 | 58.6750 | 22.6898 | 28.4 |
| GLADIUS | behavioral | 3/3 | 3/3 | [0.000, 2.367, 0.000] | - | 0.2208 | 5.6228 | 6.7224 | 21.6122 | 49.9853 | 25.6 |
| BC | behavioral | 3/3 | 3/3 | not in theta gauge (320 values) | - | 0.1298 | 2.1161 | 2.1524 | 48.9540 | 24.3452 | 0.2 |
| NFXP | structural | 3/3 | 3/3 | [-15615.134, -3419.303, -9623.201] | 10826.8304 | 0.0004 | 0.0029 | 0.0033 | 15.1643 | 0.0512 | 12.0 |
| CCP | structural | 3/3 | 3/3 | [-88.773, 87.366, -136.175] | 106.1060 | 0.0081 | 0.0028 | 0.0032 | 14.1712 | 0.0495 | 3.1 |
| MPEC | structural | 3/3 | 3/3 | [-26.021, 7.939, -19.005] | 20.1388 | 0.0003 | 0.0027 | 0.0031 | 11.2165 | 0.0491 | 8.5 |
| UFXP | structural | 3/3 | 0/3 | [-2.333, 2.456, 3.726] | 6.1309 | 0.0388 | 30.4731 | 30.5956 | 23.6958 | 27.0447 | 0.1 |

Param RMSE is the structural family only (recovered theta vs true, same gauge). Recovered params are shown only when the estimator's parameter vector lives in the data-generating gauge; a tabular reward or a choice-probability table is labeled rather than printed, because comparing it to theta entry by entry would be meaningless. Policy TV is total-variation distance from the true-parameter policy. Conv is the converged flag reported by the estimator itself; a conservative flag can read False while the recovered policy is accurate, so read it next to Policy TV, not alone. Regret is welfare loss (lower is better): `base` is the observed world; `A` payoff shift, `B` transition change, `C` action penalty. Estimators that recovered a reward in the linear feature gauge re-solve it under each intervention and adapt. Large Type C regret has two distinct routes: estimators with no reward in that gauge keep their frozen policy and cannot adapt, and an estimator that transfers a badly scaled reward adapts to the wrong world.

The headline reading is the gap between the Policy TV and Param RMSE columns for the structural family: near-perfect behavior with parameter estimates orders of magnitude from the truth. That is not estimation error, it is structural non-identification, and the design line above the table says why: the raw feature design has full rank, but the action-contrast design - the only thing choice data can ever identify - has rank 1, because the step-penalty and distance features take the same value for every action at a state and difference out of every choice probability. Any parameter vector on that two-dimensional ridge produces the same behavior, so each estimator reports an arbitrary ridge point. The practical lesson: check the rank of the action-differenced features before estimating, not just the raw design.

One regret caveat specific to this page: transitions here are deterministic, so the Type B intervention (replace the dynamics with a random sparse world) is a stark change rather than a perturbation; read Type B as a stress test of reward transferability under completely new dynamics, not as a local robustness check.

## Code used

The exact construction for each estimator (configs are modest defaults with documented fixes, not tuned per cell):

### MaxEnt-IRL

The Ziebart tradition this environment comes from. Matches discounted feature counts; with two of three features state-only, most of its objective is insensitive to the choice contrasts the data carry.

```python
def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    # Action-dependent features: the reward here depends on where the action
    # leads (terminal indicator), not on the state alone. Adaptive per-
    # parameter steps (Adam) handle the mixed feature scales.
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             learning_rate=0.05, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### MCE-IRL

Causal maximum-entropy IRL. Two of its three reward directions are unidentified here (state-only features), so its gradient ascent can drift far along them; in one replication of three the resulting policy collapsed outright. Read the per-rep records, not just the mean.

```python
def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
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

### AIRL

reward_arg='state_action'; recovered parameters stay gauge/shaping-unidentified by design, so policy TV is the right scorecard, and here even behavior is hard: the discriminator sees mostly corridor states.

```python
def _run_airl(env, panel):
    from econirl.estimation import AIRLConfig, AIRLEstimator

    est = AIRLEstimator(config=AIRLConfig(reward_type="linear", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### IQ-Learn

q_type='linear' ties its Q function to the same feature basis, so it inherits the contrast-rank problem on top of thin coverage.

```python
def _run_iq_learn(env, panel):
    from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

    est = IQLearnEstimator(config=IQLearnConfig(q_type="linear", divergence="chi2",
                                                alpha=3.0, max_iter=2000, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)
```

### f-IRL

Recovers a tabular reward, one value per state-action pair, which does not depend on the deficient feature basis at all - the strongest behavioral score on this page.

```python
def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    # fkl (bounded gradient) with the estimator's default reward clip; the
    # chi2 ratio gradient is unbounded on near-deterministic experts.
    est = FIRLEstimator(f_divergence="fkl", lr=0.2, max_iter=400, reward_clip=10.0,
                        verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### GLADIUS

Neural Q and expected-value networks; tracks behavior where data exists.

```python
def _run_gladius(env, panel):
    from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

    est = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=300, q_hidden_dim=128,
                                                v_hidden_dim=128, q_lr=1e-4, v_lr=1e-4,
                                                patience=60, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### BC

Behavioral cloning; matches observed choices where data exists, falls back to uniform where it does not, and recovers no reward.

```python
def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### NFXP

Structural contrast: exact MLE. It reproduces the true policy almost perfectly while reporting parameters far from the truth - the likelihood is flat along the two state-only feature directions, so the parameter numbers are arbitrary points on a ridge, not estimation error.

```python
def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### CCP

Structural contrast. Same flat likelihood as NFXP, plus inverted choice probabilities estimated from a concentrated state distribution.

```python
def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### MPEC

Structural contrast: constrained MLE on the same ridge.

```python
def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    est = MPECEstimator(config=MPECConfig(solver="slsqp", max_iter=200, constraint_tol=1e-6),
                        compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

### UFXP

Unnested fixed point (Bray; Oguz and Bray 2026) with optimal weighting (OUFXP). Its moment system is built from exactly the action-contrast features, so the rank-1 contrast design leaves two directions to the minimum-norm solution; behavior stays close, parameters are pinned only in the identified direction.

```python
def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point with optimal weighting (OUFXP). Conditions
    # are scored only at visited states and the optimal weights downweight
    # thin states by their sample share, which is the interesting behavior on
    # this concentrated-coverage grid.
    est = UFXPEstimator(weights="optimal", verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)
```

## Reproduce

```bash
python scripts/sim_taxi_gridworld.py                 # run + write JSON
python scripts/sim_taxi_gridworld.py --page          # regenerate this page
python scripts/sim_taxi_gridworld.py --verify        # re-derive the table from JSON
```

Raw facts: `validation/results/sim_taxi_gridworld.json`. Counterfactual regret follows the package Type A (payoff shift), Type B (transition change), Type C (action penalty) taxonomy; regret = initial_distribution . (oracle_value - estimated_value), lower is better. Estimators with a recovered reward re-solve it under each intervention (transfer); estimators without one keep their fixed policy (cannot adapt).

Excluded from this run: SEES (its spline value basis is built for an ordered 1-D state index; a 2-D grid breaks that geometry, so running it here would be a misspecification by construction); NNES, TD-CCP (the structural contrast is carried by NFXP/CCP/MPEC/UFXP here; the full structural roster runs on the bus engine and abstract MDP pages); MMP, GAIL, GCL, DeepMaxEnt-IRL, Bayesian-IRL (dropped from the study's rosters by scope decision (MMP and GAIL also failed a 20-30 minute single-fit budget on the bus engine page)).