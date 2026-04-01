# Design: Three New Datasets (Citibike, Supermarket, Entry/Exit)

## Context

econirl has 9 environments and ~15 dataset loaders covering maintenance, healthcare, transportation, labor, consumer, and IRL benchmark domains. Three gaps remain in the case study library: urban mobility (bikeshare), retail IO (pricing and inventory), and firm dynamics (entry/exit). Adding Citibike, Aguirregabiria (1999) supermarket, and Abbring-Klein entry/exit fills these gaps with minimal new infrastructure.

## 1. Citibike (Two Loaders, Two Environments)

### 1a. Route Choice: `CitibikeRouteEnvironment` + `load_citibike_route()`

**DDC framing.** Each trip is a destination choice given an origin station and time of day. States are station clusters crossed with time-of-day buckets. Actions are destination clusters.

**State space.** K-means clustering on station (lat, lng) coordinates produces ~20 station clusters. Cross with 4 time-of-day buckets (morning, midday, evening, night) for ~80 states total.

**Action space.** 20 destination clusters (same as station clusters). Self-loops allowed (short trips within a cluster).

**Features (per state-action pair).** Three features: (1) Euclidean distance from origin cluster centroid to destination cluster centroid, (2) destination cluster popularity (fraction of all trips ending there), (3) time-of-day indicator for peak hours.

**Transition dynamics.** After arriving at destination cluster d at time t, the next state is (d, t') where t' advances by one bucket with some probability of staying in the same bucket (short vs long dwell times). Transitions estimated from data.

**Data pipeline.** Download script pulls one month of Citibike Parquet from S3, clusters stations, discretizes time, and saves a processed CSV (~50K trips). Loader reads the processed file.

**Files to create:**
- `src/econirl/environments/citibike_route.py` -- `CitibikeRouteEnvironment(DDCEnvironment)`
- `src/econirl/datasets/citibike_route.py` -- `load_citibike_route()`, `get_citibike_route_info()`
- `scripts/download_citibike.py` -- downloads and preprocesses one month from S3
- `examples/citibike-route/run_estimation.py`
- `docs/examples/citibike_route.rst`

### 1b. Usage Frequency: `CitibikeUsageEnvironment` + `load_citibike_usage()`

**DDC framing.** Panel of members over days. Each day a member decides whether to ride or not. This is a labor supply and usage frequency DDC problem, similar to Buchholz's taxi driver stopping decision.

**State space.** Cross day-of-week type (weekday vs weekend, 2 buckets) with recent usage intensity (rides in last 7 days, binned into 4 buckets: 0, 1-2, 3-5, 6+). Total: 8 states.

**Action space.** 2 actions: ride or not ride.

**Features (per state-action pair).** Three features: (1) weekend indicator, (2) recent usage intensity (normalized), (3) ride cost indicator (1 for ride action, 0 for no-ride).

**Transition dynamics.** Recent usage bucket transitions deterministically based on action: riding increments the 7-day count, not riding decrements it. Day-of-week transitions are deterministic (weekday to weekday with weekend transitions on Friday/Monday).

**Data pipeline.** Same download script as route choice. Aggregates trips per member per day, builds the panel. Requires member_casual == "member" filter for panel identification.

**Files to create:**
- `src/econirl/environments/citibike_usage.py` -- `CitibikeUsageEnvironment(DDCEnvironment)`
- `src/econirl/datasets/citibike_usage.py` -- `load_citibike_usage()`, `get_citibike_usage_info()`
- `examples/citibike-usage/run_estimation.py`
- `docs/examples/citibike_usage.rst`

## 2. Aguirregabiria (1999) Supermarket: `SupermarketEnvironment` + `load_supermarket()`

**DDC framing.** A retailer manages pricing and inventory for products. Each period the retailer sets a price (markup) and decides whether to place an order. Inventories deplete through sales and replenish through orders.

**State space.** Inventory bin (5 levels: very low, low, medium, high, very high) crossed with lagged price bin (3 levels: discount, regular, premium). Total: 15 states.

**Action space.** Price level (3: discount, regular, premium) crossed with order decision (2: order, no order). Total: 6 actions.

**Features (per state-action pair).** Four features: (1) holding cost (proportional to inventory level), (2) markup (price minus wholesale, varies by price action), (3) stockout indicator (1 if inventory is very low and no order), (4) price change indicator (1 if price action differs from lagged price state).

**Transition dynamics.** Inventory evolves as inventory_next = inventory_current minus sales (stochastic, depends on price) plus order quantity (if ordered). Demand is higher at lower prices. Lagged price transitions deterministically to the chosen price action. Transitions estimated from the real data.

**Data pipeline.** Download the Stata .dta files from Aguirregabiria's website, convert to CSV using pandas, discretize inventory and prices, and bundle the processed CSV in the package. Environment calibrated from the paper's structural estimates (Table 3).

**Files to create:**
- `src/econirl/environments/supermarket.py` -- `SupermarketEnvironment(DDCEnvironment)`
- `src/econirl/datasets/supermarket.py` -- `load_supermarket()`, `get_supermarket_info()`
- `src/econirl/datasets/supermarket_data.csv` -- bundled processed data (~1-2 MB)
- `examples/supermarket/run_estimation.py`
- `docs/examples/supermarket.rst`

## 3. Abbring-Klein Entry/Exit: `EntryExitEnvironment` + `load_entry_exit()`

**DDC framing.** A firm observes a market profitability state and decides whether to be active (enter or stay) or inactive (exit or stay out). Entry and exit have sunk costs. This is the Dixit (1989) model as implemented in the Abbring-Klein teaching package.

**State space.** Market profit bin (10 levels, evenly spaced) crossed with incumbent status (2: active, inactive). Total: 20 states.

**Action space.** 2 actions: be inactive (action 0) or be active (action 1).

**Features (per state-action pair).** Four features: (1) profit flow (profit_bin value times active indicator), (2) entry cost indicator (1 if entering: was inactive and choosing active), (3) exit cost indicator (1 if exiting: was active and choosing inactive), (4) fixed operating cost indicator (1 if active).

**True parameters.** Default values calibrated to produce interesting entry/exit dynamics with hysteresis:
- `profit_slope`: 1.0 (sensitivity of profit to market state)
- `entry_cost`: -2.0 (sunk cost of entering)
- `exit_cost`: -0.5 (sunk cost of exiting)
- `operating_cost`: -0.5 (per-period fixed cost of being active)

**Transition dynamics.** Market profit follows a first-order Markov chain with persistence (AR(1)-like discretization). Incumbent status transitions deterministically based on the action chosen.

**Data pipeline.** Purely synthetic. The loader creates the environment, solves for the optimal policy, and simulates panels using `simulate_panel()`. No external data needed.

**Files to create:**
- `src/econirl/environments/entry_exit.py` -- `EntryExitEnvironment(DDCEnvironment)`
- `src/econirl/datasets/entry_exit.py` -- `load_entry_exit()`, `get_entry_exit_info()`
- `examples/entry-exit/run_estimation.py`
- `docs/examples/entry_exit.rst`

## Registration

All new loaders and environments get registered in their respective `__init__.py` files. The docs index at `docs/examples/index.rst` gets updated with new entries under appropriate domain categories.

## Hero Images

Each docs page needs a hero image in `docs/_static/`. These will be domain-appropriate illustrations (bikeshare station, supermarket shelf, factory/market entry).

## Verification

For each dataset, the example script should:
1. Load data (or generate synthetic panel)
2. Run at least two estimators (NFXP + one IRL estimator)
3. Print `etable()` comparison
4. Run post-estimation diagnostics
5. Build docs locally with `python3 -m sphinx -b html docs docs/_build/html` and verify all three new pages render

## File Summary

| Component | Citibike Route | Citibike Usage | Supermarket | Entry/Exit |
|-----------|---------------|----------------|-------------|------------|
| Environment | `environments/citibike_route.py` | `environments/citibike_usage.py` | `environments/supermarket.py` | `environments/entry_exit.py` |
| Loader | `datasets/citibike_route.py` | `datasets/citibike_usage.py` | `datasets/supermarket.py` | `datasets/entry_exit.py` |
| Bundled data | No (download script) | No (download script) | Yes (~1-2 MB CSV) | No (synthetic) |
| Download script | `scripts/download_citibike.py` (shared) | (shared with route) | Manual download + convert | N/A |
| Example | `examples/citibike-route/run_estimation.py` | `examples/citibike-usage/run_estimation.py` | `examples/supermarket/run_estimation.py` | `examples/entry-exit/run_estimation.py` |
| Docs page | `docs/examples/citibike_route.rst` | `docs/examples/citibike_usage.rst` | `docs/examples/supermarket.rst` | `docs/examples/entry_exit.rst` |
| Hero image | Yes | Yes | Yes | Yes |
