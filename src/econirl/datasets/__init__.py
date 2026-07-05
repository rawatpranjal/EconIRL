"""
Built-in datasets for econirl.

This module provides loaders for real-world datasets used in dynamic discrete
choice and inverse reinforcement learning research. Synthetic data generation
lives on the environment classes via env.generate_panel().

Real DDC Datasets:
- Rust (1987): Bus engine replacement
- SCANIA Component X (IDA 2024): Heavy truck component replacement
- Keane & Wolpin (1994): Career decisions
- Aguirregabiria (1999): Supermarket pricing/inventory
- ICU-Sepsis: Clinical treatment decisions (abstracted MDP)

Real IRL / Trajectory Datasets:
- T-Drive: Beijing taxi GPS trajectories
- GeoLife: Human mobility GPS trajectories
- Stanford Drone: Campus pedestrian/cyclist trajectories
- ETH/UCY: Pedestrian trajectory benchmark
- NGSIM: Highway lane-change vehicle trajectories
- Shanghai: Taxi route-choice on road network

Real Sequential Choice Datasets:
- Trivago (2019): Hotel search sessions
- Foursquare: Venue check-in sequences

Hybrid Datasets (real data with synthetic fallback):
- Citibike Route: Station-to-station destination choice
- Citibike Usage: Daily ride/no-ride member panel
"""

# Real DDC datasets
# Hybrid datasets (real data + synthetic fallback)
from econirl.datasets.citibike_route import get_citibike_route_info, load_citibike_route
from econirl.datasets.citibike_usage import get_citibike_usage_info, load_citibike_usage
from econirl.datasets.equipment_replacement import load_equipment_replacement
from econirl.datasets.eth_ucy import get_eth_ucy_info, load_eth_ucy
from econirl.datasets.geolife import get_geolife_info, load_geolife
from econirl.datasets.icu_sepsis import get_icu_sepsis_info, load_icu_sepsis, load_icu_sepsis_mdp
from econirl.datasets.keane_wolpin import get_keane_wolpin_info, load_keane_wolpin

# Real-world DDC/IRL datasets
from econirl.datasets.ngsim import get_ngsim_info, load_ngsim
from econirl.datasets.occupational_choice import load_occupational_choice
from econirl.datasets.rdw_scrappage import get_rdw_scrappage_info, load_rdw_scrappage
from econirl.datasets.robinson_crusoe import get_robinson_crusoe_info, load_robinson_crusoe
from econirl.datasets.rust_bus import load_rust_bus, rust_bus_reward_spec
from econirl.datasets.scania import get_scania_info, load_scania
from econirl.datasets.shanghai_route import (
    build_edge_features,
    build_state_action_features,
    build_transition_matrix,
    load_shanghai_network,
    load_shanghai_route,
    load_shanghai_trajectories,
)
from econirl.datasets.stanford_drone import get_stanford_drone_info, load_stanford_drone
from econirl.datasets.supermarket import get_supermarket_info, load_supermarket
from econirl.datasets.taxi_gridworld import get_taxi_gridworld_info, load_taxi_gridworld

# Real IRL / trajectory datasets
from econirl.datasets.tdrive import get_tdrive_info, load_tdrive
from econirl.datasets.trivago_search import (
    build_trivago_features,
    build_trivago_mdp,
    build_trivago_panel,
    build_trivago_transitions,
    get_trivago_info,
    load_trivago_search,
    load_trivago_sessions,
)

# Soft aliases so the curated dataset names used in the JSS paper line
# up with the existing loaders. The original names remain available.
load_rust_small = load_rust_bus
load_ziebart_small = load_taxi_gridworld

from econirl.datasets.shapeshifter import (
    get_shapeshifter_info,
    load_shapeshifter,
)

__all__ = [
    # Real DDC Datasets
    "load_rust_bus",
    "rust_bus_reward_spec",
    # Legacy synthetic generators (pending conversion to environments)
    "load_occupational_choice",
    "load_robinson_crusoe",
    "get_robinson_crusoe_info",
    "load_equipment_replacement",
    "load_keane_wolpin",
    "get_keane_wolpin_info",
    "load_rdw_scrappage",
    "get_rdw_scrappage_info",
    "load_scania",
    "get_scania_info",
    "load_supermarket",
    "get_supermarket_info",
    # Healthcare Datasets
    "load_icu_sepsis",
    "load_icu_sepsis_mdp",
    "get_icu_sepsis_info",
    # Hybrid Datasets
    "load_citibike_route",
    "get_citibike_route_info",
    "load_citibike_usage",
    "get_citibike_usage_info",
    # IRL Datasets
    "load_tdrive",
    "get_tdrive_info",
    "load_geolife",
    "get_geolife_info",
    "load_stanford_drone",
    "get_stanford_drone_info",
    "load_eth_ucy",
    "get_eth_ucy_info",
    # Real-world DDC/IRL Datasets
    "load_ngsim",
    "get_ngsim_info",
    # Benchmark Datasets
    "load_taxi_gridworld",
    "get_taxi_gridworld_info",
    # Shanghai route-choice
    "load_shanghai_network",
    "load_shanghai_route",
    "load_shanghai_trajectories",
    "build_transition_matrix",
    "build_edge_features",
    "build_state_action_features",
    # Trivago hotel search
    "load_trivago_search",
    "load_trivago_sessions",
    "build_trivago_mdp",
    "build_trivago_panel",
    "build_trivago_features",
    "build_trivago_transitions",
    "get_trivago_info",
    # Curated dataset aliases
    "load_rust_small",
    "load_ziebart_small",
    # Shape-shifter for code-vs-paper alignment benchmarks
    "load_shapeshifter",
    "get_shapeshifter_info",
]
