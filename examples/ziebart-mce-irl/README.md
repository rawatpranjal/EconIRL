# Ziebart MCE IRL Controlled Gridworld Study

Reproducible gridworld evidence for Ziebart-style maximum causal entropy IRL.
This is not an exact replication of Ziebart's taxi-route paper table; that
table requires the original route data or an independently reproducible
benchmark.

## Papers

- Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). "Maximum Entropy Inverse Reinforcement Learning." AAAI.
- Ziebart, B. D. (2010). "Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy." PhD Thesis, CMU.

## Scripts

- `run_gridworld.py` - Main controlled gridworld study. Runs MCE IRL and MaxEnt IRL on three synthetic gridworld reward cases.
- `ziebart_mce_irl_replication.py` - Smaller legacy gridworld diagnostic.

## Usage

```bash
# Default: 5x5 grid, 2000 trajectories
python run_gridworld.py

# Reference run used in the replication record
python run_gridworld.py --grid-size 5 --n-traj 500 --n-periods 30

# Larger grid with less data
python run_gridworld.py --grid-size 8 --n-traj 200
```

## Key Results

IRL rewards are identified only up to additive constants and multiplicative scale (Kim et al. 2021), so we evaluate on:

- **Cosine similarity** of recovered reward direction
- **Policy accuracy** (argmax agreement with true policy)
- **KL divergence** from true to recovered policy
- **Feature matching** (||empirical - expected features||)
