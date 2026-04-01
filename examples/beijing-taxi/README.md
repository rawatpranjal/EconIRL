# Beijing Taxi: MCE IRL + NFXP on T-Drive Data

Estimates taxi driver preferences from real Beijing GPS data using inverse reinforcement learning and structural estimation.

## Data

Uses the [T-Drive dataset](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) — 10,357 taxi GPS trajectories from Beijing (Feb 2008). Raw data should be in `data/raw/tdrive/`.

## Pipeline

1. **Load** GPS traces from T-Drive `.txt` files
2. **Filter** to central Beijing bounding box
3. **Discretize** into an NxN grid (default 15x15 = 225 states)
4. **Infer actions** (North/South/East/West/Stay) from consecutive cells
5. **Split** into trajectories on 30-minute gaps
6. **Estimate** transition matrices from observed transitions
7. **Run** MCE IRL and NFXP to recover driving preferences

## Features

- `step_cost`: -1 for movement, 0 for staying
- `is_stay`: 1 if staying in place
- `dist_to_center`: Normalized distance to grid center
- `northward`: Directional bias for north/south
- `eastward`: Directional bias for east/west

## Usage

```bash
# Default: 50 taxis, 15x15 grid
python run_estimation.py

# More data, finer grid
python run_estimation.py --n-taxis 200 --grid-size 20

# Save results
python run_estimation.py --save-results
```

## Dataloader

Also available as a reusable dataloader:

```python
from econirl.datasets.tdrive_panel import load_tdrive_panel
data = load_tdrive_panel(n_taxis=50, grid_size=15)
panel, transitions = data["panel"], data["transitions"]
```
