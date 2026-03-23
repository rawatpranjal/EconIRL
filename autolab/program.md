# AutoLab Research Program — econirl Hyperparameter Exploration

## Objective

Improve Tier 3-4 estimators through hyperparameter tuning. All estimators use the MultiComponentBus benchmark (K=1) evaluated via the `pct_optimal` metric (baseline-normalized: 0% = random, 100% = optimal).

## Baselines (default hyperparameters, n_states=20, seed=42)

| Tier | Estimator    | pct_optimal | Status      |
|------|-------------|-------------|-------------|
| 1    | NFXP        | 100.0%      | Converged   |
| 1    | CCP         | 100.0%      | Converged   |
| 1    | MCE IRL     | 100.0%      | Converged   |
| 2    | TD-CCP      | 99.9%       | Converged   |
| 2    | GLADIUS     | 99.8%       | Converged   |
| 2    | Max Margin  | 98.3%       | Converged   |
| 3    | AIRL        | 79.9%       | Converged   |
| 3    | MaxEnt IRL  | 47.3%       | Converged   |
| 4    | GCL         | 36.3%       | Converged   |
| 4    | GAIL        | 36.1%       | Converged   |

## Search Space

### GAIL (baseline 36.1%)
- discriminator_type: ["linear", "mlp"]
- max_rounds: [100, 200, 500, 1000]
- discriminator_lr: [0.001, 0.005, 0.01, 0.02, 0.05]
- discriminator_steps: [1, 3, 5, 10, 20]
- compute_se: false

### AIRL (baseline 79.9%)
- reward_type: ["linear", "mlp"]
- max_rounds: [100, 200, 500, 1000]
- reward_lr: [0.001, 0.005, 0.01, 0.02, 0.05]
- discriminator_steps: [1, 3, 5, 10, 20]
- compute_se: false

### GCL (baseline 36.3%)
- max_iterations: [100, 300, 500, 1000]
- cost_lr: [1e-4, 5e-4, 1e-3, 5e-3]
- embed_dim: [8, 16, 32, 64]
- hidden_dims: [[32,32], [64,64], [32,32,32]]
- importance_clipping: [3.0, 5.0, 10.0]
- n_sample_trajectories: [100, 200, 500]
- normalize_reward: [true, false]

### MaxEnt IRL (baseline 47.3%)
- inner_solver: ["value", "policy"]
- inner_max_iter: [2000, 5000, 10000]
- inner_tol: [1e-8, 1e-10]
- outer_max_iter: [100, 300, 500, 1000]
- compute_hessian: false

### GLADIUS (transfer robustness)
- bellman_penalty_weight: [0.01, 0.1, 0.5, 1.0]
- max_epochs: [200, 500, 1000]
- q_hidden_dim / v_hidden_dim: [16, 32, 64]
- q_num_layers / v_num_layers: [2, 3]
- weight_decay: [1e-4, 1e-3, 1e-2]
- batch_size: [128, 256, 512]
- compute_se: false

### DGP variations
- n_states: [10, 20, 30, 50]
- n_agents: [50, 100, 200, 500]
- discount_factor: [0.95, 0.99]

## Budget

- Max experiments: 50
- Wall-clock limit: 4 hours
- Per-experiment timeout: 600 seconds

## Ground Truth Criteria

Per-estimator success thresholds at each difficulty level (toy/standard/hard):

| Estimator   | Toy pct_optimal | Toy param_rmse | Std pct_optimal | Hard pct_optimal |
|-------------|-----------------|----------------|-----------------|------------------|
| NFXP        | 99.5%           | 0.15           | 99.0%           | 95.0%            |
| CCP         | 99.5%           | 0.15           | 99.0%           | 95.0%            |
| MCE IRL     | 99.0%           | 0.20           | 98.0%           | 90.0%            |
| Max Margin  | 95.0%           | —              | 90.0%           | 80.0%            |
| TD-CCP      | 98.0%           | 0.25           | 95.0%           | 85.0%            |
| GLADIUS     | 98.0%           | 0.25           | 95.0%           | 85.0%            |
| AIRL        | 70.0%           | —              | 50.0%           | 30.0%            |
| MaxEnt IRL  | 60.0%           | —              | 40.0%           | 20.0%            |
| GAIL        | 50.0%           | —              | 30.0%           | 10.0%            |
| GCL         | 50.0%           | —              | 30.0%           | 10.0%            |

Difficulty levels by n_states: toy (<=10), standard (11-30), hard (>30).

## Validation Strategy

1. **Toy baselines first**: Run each estimator on toy DGP (n_states=5, gamma=0.95) to validate correctness before spending budget on harder problems
2. **Tune on standard**: Once toy passes, tune hyperparameters on standard DGP (n_states=20, gamma=0.99)
3. **Stress-test winners**: Validate best configs on hard DGP (n_states=50, gamma=0.99)

## Strategy

1. Start with GAIL (largest gap to close, fast to run)
2. Explore discriminator architecture and learning rate first
3. After GAIL, move to GCL, then MaxEnt IRL, then AIRL
4. Validate top-3 configs per estimator across seeds [42, 123, 456]
5. Try DGP variations (n_states, discount_factor) on winners
