# Real-World Dataset Survey for econirl Benchmarking

## Overview

This report maps real-world datasets to IRL/DDC algorithms based on a comprehensive review of all papers in the repository and the `rawatpranjal/data-for-irl` dataset list. The goal is to identify which datasets could serve as benchmarks for econirl's estimators beyond the current synthetic-only evaluation.

## Paper Coverage

- **68 non-empty PDFs** read across `papers/foundational/`, `papers/literature/`, `papers/ore/`
- **5 docling'd papers** in `papers/article_pranjal/new_papers_to_add/`
- **37-page Rawat-Rust survey** (the main article) covering DDC and IRL literature comprehensively
- **~50+ empty placeholder PDFs** in `papers/literature/` (not readable)

---

## Datasets Found in Papers

### Tier 1: Canonical DDC Datasets (Already in econirl)

| Dataset | Paper | Type | States | Actions | In econirl |
|---------|-------|------|--------|---------|------------|
| **Rust Bus Engine Replacement** | Rust (1987) | Maintenance DDC | 90 mileage bins | 2 (keep/replace) | `load_rust_bus()` |
| **Career Decisions (NLSY)** | Keane & Wolpin (1997) | Labor DDC | Multi-dim (edu x exp x age) | 4-5 (school/work types) | `load_keane_wolpin()` |

### Tier 2: Real-World Datasets Used in Papers (Public, Suitable for econirl)

| Dataset | Paper | Size | Type | Availability |
|---------|-------|------|------|-------------|
| **Pittsburgh taxi GPS** | Ziebart et al. (2008) | 25 drivers, 13K trips, 12 weeks | Route choice IRL | Public (CMU) |
| **AHEAD elderly survey** | Kaji et al. (2023, Econometrica) | 3,259 retirees, 1994-2006 | Consumption/saving DDC | Public (HRS/AHEAD) |
| **PROGRESA school subsidy** | Todd & Wolpin (2006, AER) | Mexican households | Schooling/fertility DDC | Public |
| **Retirement (HRS)** | Rust & Phelan (1997) | US retirees | Labor/retirement DDC | Public (HRS) |
| **Patent renewal** | Pakes (1986) | Patent holders | Renewal DDC | Public (USPTO) |

### Tier 3: Real-World Datasets in Papers (Limited Availability)

| Dataset | Paper | Size | Type | Availability |
|---------|-------|------|------|-------------|
| **Google Maps routes** | Barnes et al. (2024) | 110M trips, 360M params | Route choice IRL | Not public |
| **Shanghai taxi trajectories** | Zhao & Liang (2023) | Taxi GPS | Route choice AIRL | Unknown |
| **Serialized fiction platform** | Lee et al. (2026) | 24K users, 290K trajectories | Content consumption AIRL | Not public |
| **Singapore travel survey** | Liang et al. (2026) | Daily activity sequences | Activity-travel DIRL | Not public |
| **Danish automobile market** | Gillingham et al. (2022) | Car ownership | Trade DDC | Not public |
| **YouTube commenting** | Hoiles et al. (2020) | User engagement | Rational inattention IRL | Unknown |
| **Professional tennis** | Anderson et al. (2025, JPE) | Serve decisions | Strategic beliefs | Partial |

### Tier 4: Simulation-Only Papers (No Real Datasets)

Most papers in the repo use simulation only:
- **IRL papers**: Ng & Russell (2000), Abbeel & Ng (2004), Ho & Ermon (2016) GAIL, Fu et al. (2018) AIRL, Garg et al. (2021) IQ-Learn — all use MuJoCo/gridworld
- **DDC-IRL bridge papers**: Zeng et al. (2022, 2024), Geng et al. (2023), Kang et al. (2025) GLADIUS — all use Rust bus simulation
- **Theoretical**: Cao et al. (2021) identifiability, Christiano et al. (2017) RLHF

---

## Datasets from data-for-irl Repository

### Best Fit for econirl (Discrete Choices, Public)

| Dataset | Why Good | Actions | States | Feasibility |
|---------|----------|---------|--------|-------------|
| **Citi Bike NYC** | Naturally discrete station choice, massive public data, seasonal patterns | N destination stations | (origin, time, weekday) | HIGH - download from citibikenyc.com |
| **NGSIM US-101** | Highway driving, widely used in IRL papers | 3-5 (lane change + speed) | (lane, speed_bin, gap) ~500-2K states | HIGH - public FHWA data |
| **GPS with Transport Mode Labels** | Mode choice is classic discrete choice (walk/bike/drive/bus) | 4-6 transport modes | (location, time, context) | HIGH - from GeoLife |
| **Career Decisions (K&W)** | Already in econirl | 4-5 | Multi-dim | Already implemented |
| **Bus Engine Replacement** | Already in econirl | 2 | 90 | Already implemented |

### Good Fit (Need Discretization, Established in IRL)

| Dataset | Notes | Effort |
|---------|-------|--------|
| **HighD Dataset** | German highway, drone-collected, high precision. Similar to NGSIM but cleaner. | Medium - need discretization |
| **inD Dataset** | Urban intersection trajectories. Complex interactions. | Medium |
| **T-Driver (Beijing taxi)** | Already have sample in econirl. Need road network to MDP. | Medium |
| **Foursquare check-ins** | Venue choice is naturally discrete. | Medium - need state design |
| **LaDe last-mile delivery** | Routing/sequencing choices. 10M+ packages. | High - large action space |

### Control/Locomotion Benchmarks (D4RL)

| Dataset | Description | Format |
|---------|-------------|--------|
| **D4RL MuJoCo Expert** | halfcheetah, hopper, walker2d, ant expert trajectories | ~1M timesteps each |
| **D4RL Adroit/Kitchen** | Dexterous manipulation demos | Variable size |
| **TROFI benchmarks** | Quality-ranked trajectory subsets from D4RL | Filtered D4RL |

These are continuous control tasks — not directly suitable for econirl's discrete DDC framework but useful for neural estimators (TD-CCP, Deep MaxEnt).

### Autonomous Driving (Trajectory Prediction)

| Dataset | Description | Size | Availability |
|---------|-------------|------|-------------|
| **Argoverse** | Pittsburgh/Detroit trajectories with maps | 300K+ trajectories | Public |
| **nuScenes** | Boston/Singapore ego-vehicle tracks | 1K+ scenes, 40s each | Public |
| **rounD/inD** | European roundabouts/intersections | 29K/3K samples | Public |

These require discretization of continuous states/actions but are well-established IRL benchmarks.

### Poor Fit for econirl

| Dataset | Reason |
|---------|--------|
| Video engagement datasets | Not DDC problems, continuous engagement metrics |
| Gaming datasets (CS:GO, StarCraft) | Continuous state/action, different framework |
| AI4Animation | Not economic/behavioral choice |
| Ithaca365, COMPASS | Perception datasets, not behavioral |

---

## Already in econirl Datasets

| Loader | Type | Real/Synthetic | Source |
|--------|------|---------------|--------|
| `load_rust_bus()` | DDC | Real + synthetic | Rust (1987) NFXP data |
| `load_keane_wolpin()` | DDC | Synthetic sample (respy optional) | Keane & Wolpin (1997) |
| `load_tdrive()` | IRL | Synthetic sample | Microsoft Research T-Drive |
| `load_geolife()` | IRL | Synthetic sample | Microsoft Research GeoLife |
| `load_stanford_drone()` | IRL | Synthetic sample | Stanford CVGL |
| `load_eth_ucy()` | IRL | Synthetic sample | ETH/UCY pedestrian |
| `load_occupational_choice()` | DDC | Generated | Keane-Wolpin style |
| `load_robinson_crusoe()` | DDC | Generated | Pedagogical |
| `load_equipment_replacement()` | DDC | Generated | Rust-style variants |

Note: The IRL datasets (T-Drive, GeoLife, Stanford Drone, ETH/UCY) currently bundle **synthetic** samples that mimic the real data distribution. Real data must be downloaded separately.

---

## Recommended Priority for Real-World Benchmarking

### Priority 1: Easiest to integrate, highest impact

1. **Citi Bike NYC** — Naturally discrete (station choice), massive public data (millions of trips), clean CSV format. Model as: given (origin_station, hour, day_of_week), predict destination_station. Binary: ride vs no-ride at each station pair.

2. **NGSIM US-101** — Public (FHWA), canonical IRL driving benchmark. Discretize to (lane, speed_bin) states with lane-change actions. Many IRL papers use this.

3. **Rust Bus (real data)** — Already in econirl as `load_rust_bus(original=True)`. The canonical DDC benchmark. Could add more bus groups from original NFXP package.

### Priority 2: Public, good DDC fit, moderate effort

4. **AHEAD/HRS Survey** — Public longitudinal data on elderly consumption/saving decisions. Used by Kaji et al. (2023) in Econometrica. Genuine DDC problem with binary/ternary choices.

5. **D4RL Expert Trajectories** — Easy to install via pip. MuJoCo tasks provide continuous-state benchmarks for neural estimators (TD-CCP, Deep MaxEnt, GLADIUS). Standard offline IRL benchmark.

6. **Pittsburgh Taxi Routes** (Ziebart 2008 data) — The canonical MCE IRL dataset. Grid-world routing with known road network. May need to contact CMU for data.

### Priority 3: Valuable but more effort

7. **Argoverse / nuScenes** — Large driving trajectory datasets. Need ego-trajectory extraction and discretization. Standard trajectory prediction benchmarks.

8. **Foursquare NYC Check-ins** — Venue/location choice. Naturally discrete actions. Need state-space design.

9. **Transport Mode Choice** (from GeoLife labels) — Walk/bike/drive/bus choice is a natural discrete choice problem matching econirl's framework perfectly.

---

## Technical Requirements for Integration

### For any new dataset, we need:
1. **Discrete states** — Map continuous observations to state indices (< 5000 for dense transitions)
2. **Discrete actions** — Define meaningful choice set (2-10 actions)
3. **Transition matrix** — Estimate P(s'|s,a) from data, shape `(num_actions, num_states, num_states)`
4. **Feature matrix** — Define phi(s,a) for utility, shape `(num_states, num_actions, num_features)`
5. **Panel format** — Convert to `Trajectory` objects: (states, actions, next_states) tensors

### State space budget:
- < 100 states: All estimators work (NFXP, CCP, MCE IRL, MaxEnt, etc.)
- 100-1000 states: Most estimators work, MCE IRL slows down
- 1000-5000 states: Only BC, CCP, TD-CCP, GLADIUS feasible
- > 5000 states: Requires sparse transitions or function approximation

---

## Empty Placeholder Papers

The following directories contain mostly empty PDF files (git-lfs placeholders or stubs):
- `papers/literature/ddc_using_rl/` — ~40 of 56 files empty
- `papers/literature/Centaur_digital_twins/` — All 13 files empty
- `papers/literature/mdp/` — All 3 files empty
- `papers/literature/imitation_learning/` — Both files empty
- `papers/literature/miscellaneous/` — Not checked (likely empty)
- `papers/literature/reinforcement_learning/` — Not checked (likely empty)

These papers should be re-downloaded if their dataset references are needed.
