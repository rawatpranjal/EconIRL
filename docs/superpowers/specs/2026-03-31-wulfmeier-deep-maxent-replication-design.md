# Wulfmeier (2016) Deep MaxEnt IRL Replication

## Purpose

Replicate the Objectworld and Binaryworld benchmarks from Wulfmeier, Ondruska & Posner (2016) "Maximum Entropy Deep Inverse Reinforcement Learning" (arXiv:1507.04888). The paper shows that neural reward functions outperform linear MaxEnt IRL when the true reward has nonlinear feature interactions. The replication validates our MCEIRLNeural estimator against the original paper results and produces a new example for the documentation.

## Environments

### Objectworld

A 32x32 grid with 1024 states and 5 actions (N, S, E, W, Stay). Transitions are deterministic. Discount factor gamma = 0.9. C colored objects are placed randomly on the grid (default C=2, with 3 objects per color following Levine et al. 2011). The reward at each state depends on the minimum distance to objects of each color.

The reward function is: positive (+1) if within distance 3 of color 1 AND within distance 2 of color 2, negative (-1) if only within distance 3 of color 1 and not within distance 2 of color 2, and zero otherwise.

Two feature representations are provided. The continuous set has C dimensions, each equal to the minimum Euclidean distance from the state to the nearest object of that color normalized by the grid size. The discrete set has C times M binary dimensions, where each indicates whether an object of color c is within distance d for d in 1 through M.

**File:** `src/econirl/environments/objectworld.py`

Inherits from `DDCEnvironment`. Constructor takes `grid_size`, `n_colors`, `n_objects_per_color`, `feature_type` ("continuous" or "discrete"), `discount_factor`, and `seed`. Exposes `transition_matrices`, `feature_matrix`, `true_reward`, `problem_spec`, and `true_parameters` properties following the existing `GridworldEnvironment` pattern.

### Binaryworld

Same 32x32 grid structure with gamma = 0.9. Every cell is randomly assigned one of two colors (blue or red). The feature vector for each state is a binary vector of length 9 encoding the colors of the 3x3 neighborhood centered on that cell (including the cell itself). Boundary cells use zero-padding.

The reward function depends on the count of blue cells in the 3x3 neighborhood. The reward is +1 if exactly 4 of 9 cells are blue, -1 if exactly 5 of 9 are blue, and 0 otherwise.

**File:** `src/econirl/environments/binaryworld.py`

Same interface as Objectworld. Constructor takes `grid_size`, `discount_factor`, and `seed`.

### Shared Design

Both environments produce:
- `transition_matrices`: shape (5, S, S) deterministic transitions
- `feature_matrix`: shape (S, 5, K) state-action features (same features for all actions, following the paper which uses state-only features broadcast to actions)
- `true_reward`: shape (S,) the ground-truth reward map
- A `simulate_demonstrations(n_demos, noise_fraction=0.3)` method that samples trajectories from the optimal policy with 30% random action noise, returning a Panel

Both environments share the 5-action deterministic grid transitions with the existing `GridworldEnvironment`. Extract the grid transition logic into a shared helper `_build_grid_transitions(grid_size)` in a `gridworld_utils.py` module to avoid duplication.

## Benchmark Script

**File:** `examples/wulfmeier-deep-maxent/replicate.py`

The script runs both benchmarks end to end. For each environment it generates expert demonstrations at N = {8, 16, 32, 64, 128}, runs MCEIRLNeural (DeepIRL) and MCE-IRL (linear MaxEnt), and measures the expected value difference (EVD).

### Expected Value Difference

EVD is the standard IRL evaluation metric from the paper. It measures the suboptimality of the policy derived from the learned reward compared to the optimal policy under the true reward. Specifically: EVD = V*(s0; r_true) - V_pi_learned(s0; r_true), averaged over a uniform distribution of start states. Lower EVD means better reward recovery.

### Experimental Protocol

For each environment and each value of N:
1. Generate the environment with a fixed seed
2. Sample N expert demonstrations (30% noise)
3. Run MCEIRLNeural with a 2-hidden-layer MLP (64 units, ReLU) and AdaGrad optimizer
4. Run linear MCE-IRL as the baseline
5. Compute EVD on the training environment
6. Generate 5 transfer environments (new random object placements but same reward structure rules) and compute EVD on each
7. Report mean and standard deviation across 5 random seeds

### Output

The script prints a results table showing EVD for each estimator at each N, for both training and transfer settings. It also saves a JSON file with all results for plotting.

### Network Architecture

Following the paper: FCNN with 2 hidden layers, ReLU activations, width-1 convolutional filters (equivalent to per-state MLP). The input is the state feature vector and the output is a scalar reward. AdaGrad optimizer with learning rate 0.01. Training for 200 epochs with early stopping.

## Reuse

- `MCEIRLNeural` from `src/econirl/estimators/mceirl_neural.py` implements Algorithm 1
- `MCEIRLEstimator` from `src/econirl/estimation/mce_irl.py` for the linear baseline
- `SoftBellmanOperator` and `policy_iteration` from `src/econirl/core/` for solving the MDP
- `simulate_panel` pattern from `src/econirl/simulation/synthetic.py` for demo generation
- `DDCEnvironment` base class from `src/econirl/environments/` for environment structure
- `GridworldEnvironment._build_transitions()` logic for deterministic grid transitions

## Tests

**File:** `tests/test_objectworld.py`

Verify environment properties: transition matrix shape and row normalization, feature matrix shape, reward values are in {-1, 0, +1}, demonstration generation produces valid trajectories.

**File:** `tests/test_binaryworld.py`

Same structure. Additionally verify that the 3x3 neighborhood features are correct for known cell configurations.

**File:** `tests/test_wulfmeier_benchmark.py` (marked @slow)

Run a small-scale version (8x8 grid, N=16 demos) and verify that MCEIRLNeural achieves lower EVD than linear MCE-IRL on Binaryworld (where the nonlinear advantage should be clear).

## Verification

1. Run `python examples/wulfmeier-deep-maxent/replicate.py` end to end
2. Check that EVD curves show DeepIRL matching or beating MaxEnt on Objectworld and clearly beating it on Binaryworld, consistent with Figures 5 and 6 in the paper
3. Run `python -m pytest tests/test_objectworld.py tests/test_binaryworld.py tests/test_wulfmeier_benchmark.py -v`
4. All tests pass with no regressions in existing test suite
