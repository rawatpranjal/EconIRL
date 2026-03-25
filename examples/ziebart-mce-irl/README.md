# Ziebart MCE IRL Replication

Replication of experiments from Ziebart et al. (2008, 2010) using the `econirl` package.

## Papers

- Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). "Maximum Entropy Inverse Reinforcement Learning." AAAI.
- Ziebart, B. D. (2010). "Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy." PhD Thesis, CMU.

## Scripts

- `run_gridworld.py` — Main replication script. Runs MCE IRL and MaxEnt IRL on a gridworld, compares recovered rewards and policies.

## Usage

```bash
# Default: 5x5 grid, 100 trajectories
python run_gridworld.py

# Larger grid with more data
python run_gridworld.py --grid-size 8 --n-traj 200

# Save results to JSON
python run_gridworld.py --save-results --output-dir results
```

## Key Results

IRL rewards are identified only up to additive constants and multiplicative scale (Kim et al. 2021), so we evaluate on:

- **Cosine similarity** of recovered reward direction
- **Policy accuracy** (argmax agreement with true policy)
- **KL divergence** from true to recovered policy
- **Feature matching** (||empirical - expected features||)
