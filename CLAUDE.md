# CLAUDE.md - Project Context for Claude Code

## Project Overview

**econirl** is a Python library for structural estimation and inverse reinforcement learning (IRL) in dynamic discrete choice models. It bridges econometrics (NFXP, CCP) with machine learning (MaxEnt IRL, MCE IRL, GAIL, AIRL).

## Key Architecture

```
src/econirl/
├── core/           # Types (DDCProblem, Panel, Trajectory), Bellman operators, solvers
├── estimation/     # 10 estimators: NFXP, CCP, MaxEnt IRL, MCE IRL, Max Margin,
│   │                 GCL, TD-CCP, GLADIUS, GAIL, AIRL
│   └── adversarial/  # GAIL and AIRL implementations
├── environments/   # RustBus, MultiComponentBus, Gridworld
├── preferences/    # LinearUtility, ActionDependentReward, NeuralCost
├── inference/      # Standard errors, identification diagnostics, EstimationSummary
├── simulation/     # Synthetic data generation, Monte Carlo, counterfactuals
└── visualization/  # Policy and value function plots
```

Each subdirectory has its own CLAUDE.md with detailed interface documentation. See:
- `docs/estimator_guide.md` - **Estimator selection guide**: why each of the 9 core estimators exists, paper-backed theorems, and when to use which
- `src/econirl/core/CLAUDE.md` - Types, Bellman operator, solvers
- `src/econirl/estimation/CLAUDE.md` - All 10 estimators and base contract
- `src/econirl/environments/CLAUDE.md` - Environment base class and 3 implementations
- `src/econirl/preferences/CLAUDE.md` - Utility function protocol and LinearUtility
- `src/econirl/inference/CLAUDE.md` - SE methods, identification, EstimationSummary
- `src/econirl/simulation/CLAUDE.md` - Data generation, Monte Carlo, counterfactuals
- `tests/CLAUDE.md` - Test conventions, fixtures, running tests

## Focus Estimators (Replication Priority)

The following estimators are the core focus for real-data replication and benchmarking:

1. **NFXP-NK** — Nested Fixed Point with SA→NK polyalgorithm (Iskhakov et al. 2016). BHHH optimizer, analytical gradient via implicit differentiation. The main structural estimator.
2. **CCP** 
2. **NNES** — Neural Network Estimation of Structural models (neural version of NFXP)
7. **TD-CCP** — Temporal Difference CCP (AE paper 2022, neural approximate VI) (VERY IMP)
3. **SEES** — Simulation-based Estimation of Economic Structural models
4. **MCE-IRL** — Maximum Causal Entropy IRL (Ziebart 2010). Feature matching with soft Bellman.
5. **MCE-IRL (Deep)** — Deep MaxEnt with neural reward function (in the style of imitation learning)
6. **GLADIUS** — Model-free DDC estimation (Q-network + EV-network)
8. **AIRL** — Adversarial IRL (Fu et al. 2018). Reward recovery via discriminator.

Each estimator should be validated on at least one real-data replication (Rust bus, Keane-Wolpin, NGSIM, etc.) before being considered production-ready.

## Critical Implementation Details

### MCE IRL Expected Features (IMPORTANT)
The `_compute_expected_features()` method in `mce_irl.py` uses occupancy measures (state visitation frequencies under the current policy) following Ziebart (2010) Algorithm 1 and the `imitation` library. The E_π term is: `E_π[φ] = Σ_s D_π(s) Σ_a π(a|s) φ(s,a,k)` where D_π is computed via the forward pass. This correctly handles both action-dependent and state-only features. Falls back to empirical-state iteration only when transitions are unavailable.

### Parameter Identification in IRL
- Rewards are only identified up to constants (Kim et al. 2021, Cao & Cohen 2021)
- For well-conditioned optimization, normalize parameters to unit norm
- Normalize features to [-1, 1] range
- Rust bus parameters (0.001, 3.0) have poor scaling - consider normalization

### Transition Matrix Conventions
- Estimators expect: `(n_actions, n_states, n_states)` i.e., `transitions[a, s, s']`
- Some internal code uses: `(n_states, n_actions, n_states)` i.e., `transitions[s, a, s']`
- Always check the expected format when passing transitions

## Testing Commands

```bash
# Run quick tests (skip slow)
python3 -m pytest tests/ -v -m "not slow"

# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_mce_irl_action_features.py -v

# Run with coverage
python3 -m pytest tests/ --cov=econirl
```

## Common Debugging

### MCE IRL not converging
1. Check feature normalization (should be [-1, 1])
2. Check parameter initialization (don't use zeros)
3. Verify inner loop (soft VI) converges - increase `inner_max_iter` for high gamma
4. Try smaller learning rate if oscillating

### NFXP slow convergence
1. High discount factor (gamma > 0.99) requires many inner iterations
2. Consider lowering gamma for testing
3. Check that transitions are properly normalized (rows sum to 1)

## Benchmarking Philosophy

Controlled simulated experiments use in-sample, out-of-sample, and out-of-transfer evaluation. Benchmarks are modular (`ESTIMATORS` dict) so new algorithms slot in with one entry. Use high trajectory counts — the goal is not standard errors but verifying that estimators correctly recover reward parameters and produce good policies. Working examples live in `examples/`.

MCE IRL and NFXP converge to the same answer with enough data and proper features — with 2000 trajectories on a 5x5 gridworld both achieve cosine similarity 0.9999 to true parameters and identical policy performance across in-sample, out-of-sample, and all three transfer scenarios. Features MUST be action-dependent (vary across the choice set) for NFXP identification; state-only features that are identical across actions make R(s,a) constant across actions, collapsing the likelihood surface and causing parameter blowup. IRL rewards are identified only up to additive constants and scale (Kim et al. 2021), so evaluate on cosine similarity and policy quality rather than raw RMSE. MaxEnt IRL underperforms MCE IRL even in deterministic environments because it doesn't account for causal structure in the state visitation computation. For the `imitation` library comparison: econirl's infinite-horizon formulation with action-dependent features is more general; the main borrowed improvement is the direct linear solve for occupancy measures in `core/occupancy.py`.

## Key References

- Rust (1987): Optimal Replacement of GMC Bus Engines
- Ziebart (2010): Maximum Causal Entropy IRL
- Kim et al. (2021): Reward Identification in IRL
- Cao & Cohen (2021): Identifiability in IRL
