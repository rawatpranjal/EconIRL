# AutoLab Research Program — Toy MDP Autoresearch

## Objective

Achieve near-perfect performance (90-99.5% pct_optimal) for ALL 10 estimators on a tiny 5-state MDP. On a problem this small, every method should recover the optimal policy if tuned correctly. The goal is to find what hyperparameters make each estimator work flawlessly on the simplest possible problem.

## Fixed DGP (do not change)

All experiments MUST use these exact parameters:

- **n_states**: 5
- **discount_factor**: 0.95
- **n_agents**: 100
- **n_periods**: 50
- **seed**: 42

## Baselines & Aspirational Targets

| Tier | Estimator    | Baseline    | Target   |
|------|-------------|-------------|----------|
| 1    | NFXP        | ~100%       | 99.5%+   |
| 1    | CCP         | ~100%       | 99.5%+   |
| 1    | MCE IRL     | ~100%       | 99.0%+   |
| 2    | TD-CCP      | (run first) | 98.0%+   |
| 2    | GLADIUS     | (run first) | 98.0%+   |
| 2    | Max Margin  | (run first) | 95.0%+   |
| 3    | AIRL        | (run first) | 90.0%+   |
| 3    | MaxEnt IRL  | (run first) | 90.0%+   |
| 4    | GCL         | (run first) | 90.0%+   |
| 4    | GAIL        | (run first) | 90.0%+   |

## Bag of Ideas

Pick from these ideas in any order based on what you learn from results. You don't have to go in order. Focus effort where the gap to target is largest.

### Baselines & Warm-Starting (Ideas 1-4)

1. **Establish defaults** — Run each estimator with default hyperparameters on the toy DGP. This is the mandatory first step for any estimator without a baseline.
2. **CCP warm-start for GAIL** — Initialize GAIL's discriminator using empirical CCPs from data. The discriminator should start near the true policy rather than random.
3. **CCP warm-start for AIRL** — Same idea: initialize AIRL's reward network from CCP-estimated values.
4. **Oracle initialization for MaxEnt IRL** — Initialize MaxEnt IRL parameters near the true values to check if the optimizer can maintain a good solution (diagnostic for convergence basin).

### Learning Rate & Optimization (Ideas 5-8)

5. **Tiny learning rate + many rounds for GAIL** — Try lr=0.0001 with max_rounds=1000-2000. Adversarial methods often need slow, stable optimization.
6. **Tiny learning rate + many rounds for AIRL** — Same approach for AIRL: reward_lr=0.0001, max_rounds=1000-2000.
7. **Adam vs SGD for MaxEnt IRL** — If MaxEnt IRL supports different optimizers, compare Adam with lr=0.001 vs default.
8. **Learning rate warmup for GCL** — Try very small cost_lr (1e-5) with many iterations (1000+). GCL's importance sampling can be unstable at high learning rates.

### Architecture & Capacity (Ideas 9-12)

9. **Tiny discriminator for GAIL** — Use discriminator with hidden_dim=8 or 16 (not 64). A 5-state problem doesn't need a big network.
10. **Tiny reward network for AIRL** — Same: small reward network (hidden_dim=8-16) for 5 states.
11. **Tiny cost network for GCL** — embed_dim=4-8, hidden_dims=[8,8]. The cost function for 5 states is simple.
12. **Tiny networks for GLADIUS/TD-CCP** — q_hidden_dim=8-16, v_hidden_dim=8-16. Match network capacity to problem complexity.

### Regularization & Stability (Ideas 13-17)

13. **Entropy bonus for GAIL** — Add entropy regularization to the policy to prevent premature convergence to a suboptimal deterministic policy.
14. **Importance clipping for GCL** — Try importance_clipping=2.0 or 3.0 to reduce variance in GCL's importance-weighted gradients.
15. **Reward normalization for GCL** — Set normalize_reward=True to keep GCL's learned costs in a stable range.
16. **Bellman penalty tuning for GLADIUS** — Try bellman_penalty_weight in [0.01, 0.1, 0.5, 1.0]. Higher penalty enforces Bellman consistency but may over-constrain.
17. **Weight decay for neural methods** — Add weight_decay=0.01 to TD-CCP, GLADIUS to prevent overfitting on 5-state data.

### Training Duration (Ideas 18-20)

18. **Extended training for GAIL** — max_rounds=2000-5000. Maybe GAIL just needs more rounds to converge on this problem.
19. **Extended training for GCL** — max_iterations=2000-5000. Same hypothesis.
20. **Extended training for MaxEnt IRL** — outer_max_iter=1000-2000, inner_max_iter=10000. Ensure both inner and outer loops fully converge.

### Solver & Inner Loop (Ideas 21-23)

21. **Policy iteration for MaxEnt IRL** — inner_solver="policy" instead of "value". Policy iteration can converge faster for small MDPs.
22. **Tight inner tolerance** — inner_tol=1e-12 for MaxEnt IRL, MCE IRL. Ensure the inner soft VI loop converges precisely.
23. **Hybrid value/policy iteration for TD-CCP** — If TD-CCP supports solver selection, try policy iteration for the inner Bellman solve.

### Discriminator Balance (Ideas 24-25)

24. **More discriminator steps for GAIL** — discriminator_steps=10-20 per policy round. The discriminator may need to be stronger to provide useful gradients.
25. **Fewer discriminator steps for AIRL** — discriminator_steps=1-2. AIRL's structured discriminator may overfit with too many steps on 5 states.

## Budget

- **Max experiments**: 30
- **Wall-clock limit**: 2 hours
- **Per-experiment timeout**: 120 seconds

## Instructions for Claude

When proposing experiments:
- If an estimator has no baseline result yet, run Idea #1 (defaults) first
- Focus effort where the gap between current best and target is largest
- Report which `idea_number` (1-25) you are testing
- Do NOT change the DGP — all experiments use n_states=5, discount_factor=0.95
- Combine ideas if it makes sense (e.g., tiny network + low learning rate)
