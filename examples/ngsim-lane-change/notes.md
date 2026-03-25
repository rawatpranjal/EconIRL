# NGSIM US-101 Lane-Change IRL — Replication Notes

## Dataset

- **Source**: FHWA Next Generation Simulation (NGSIM) US-101 freeway, Los Angeles
- **Collection**: June 15, 2005, southbound US-101 (Hollywood Freeway)
- **Size**: 4.8M trajectory frames, 2,848 vehicles, 10Hz sampling
- **Download**: `data/raw/ngsim/us101_trajectories.csv` (814MB)

## Problem Setup

- **State**: (lane, speed_bin) — 5 lanes x 10 speed bins = **50 states**
- **Actions**: lane_left / stay / lane_right = **3 actions**
- **Discount**: beta = 0.99
- **Data used**: 500 vehicles, subsampled to 1Hz, ~79K observations

## Features (Matching Huang, Wu & Lv 2021)

5 structural features inspired by Huang et al.'s 8-feature reward:

| Feature | Definition | Action-Dependent? |
|---------|-----------|:-----------------:|
| speed | speed_bin / max (structural) | No |
| accel_cost | -mean_accel_per_state | No |
| headway_risk | -mean_exp(-THW)_per_state | No |
| collision_risk | -mean_collision_rate_per_state | No |
| **lane_change_cost** | **-1 if lane change, 0 if stay** | **Yes** |

## Identification Issue

Features 1-4 are **state-only** (identical across all 3 actions within each state). In a logit/softmax model:

    P(a|s) = exp(theta^T phi(s,a)) / sum_a' exp(theta^T phi(s,a'))

If phi(s,a,k) is the same for all actions a, theta_k cancels in the ratio and has **zero effect** on choice probabilities. These parameters lie on a flat ridge of the log-likelihood.

Only `lane_change_cost` is action-dependent and thus **identified from choice data**. All estimators agree on this parameter (~2.38).

This is the classic IRL identification problem (Kim et al. 2021, Cao & Cohen 2021): rewards are identified only up to state-dependent constants when actions don't affect state features.

## Results

| Estimator | Time | Conv | LL | lane_change_cost | speed | accel | headway | collision |
|-----------|-----:|:----:|-------:|:----------------:|------:|------:|--------:|----------:|
| **NFXP-NK** | 5.1s | Yes | -42,328 | **2.384** | 0.253 | 0.062 | 0.170 | 0.034 |
| **MCE IRL** (L-BFGS-B) | 2.4s | Yes | -42,336 | **2.386** | 0.226 | 0.048 | 0.162 | 0.071 |
| MaxEnt IRL | 7.8s | ~Yes | -42,347 | **2.380** | 0.043 | -0.011 | -0.011 | 0.023 |
| AIRL | 438s | No | -47,304 | 1.493 | 0.076 | -0.004 | -0.049 | -0.007 |

## Key Findings

### 1. DDC-IRL Equivalence Validated on Real Data
NFXP-NK and MCE IRL (L-BFGS-B) produce nearly identical:
- `lane_change_cost`: 2.384 vs 2.386 (0.1% difference)
- Log-likelihood: -42,328 vs -42,336 (8 nats, <0.02%)
- State-only features: same signs, similar magnitudes

This confirms the theoretical equivalence (Rawat-Rust survey Section 4.1): MaxEnt IRL with linear features = MLE with logit shocks.

### 2. MCE IRL Optimizer Matters
- **Adam optimizer**: 94s, did NOT converge (LL = -47,927). First-order optimizer oscillates.
- **L-BFGS-B optimizer**: 2.4s, CONVERGED (LL = -42,336). Second-order = reliable.
- The `MCEIRLConfig.optimizer` field was previously unused (bug). Now wired to scipy L-BFGS-B.

### 3. AIRL Is Harder to Converge
AIRL finds reasonable lane_change_cost (1.49) but lower magnitude and worse LL. Adversarial training inherently harder than direct likelihood optimization for tabular problems.

### 4. Comparison with Huang et al. (2021)
Huang et al. use trajectory-level MaxEnt IRL (Ziebart 2008 style, not MCE IRL):
- **Their approach**: Sample ~33 candidate polynomial trajectories, softmax over trajectory features
- **Our approach**: Tabular MCE IRL with Bellman equation over 50 discrete states
- **Agreement**: Both find speed preference positive, lane changes costly
- **Their advantage**: Continuous state/action, per-driver personalization
- **Our advantage**: Exact solution (no trajectory sampling), standard errors, DDC-IRL equivalence proof

## References

- Huang, Z., Wu, J., & Lv, C. (2021). "Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning." IEEE T-ITS.
- Ziebart, B.D. et al. (2008). "Maximum Entropy Inverse Reinforcement Learning." AAAI.
- Ziebart, B.D. (2010). "Modeling purposeful adaptive behavior with the principle of maximum causal entropy." PhD thesis, CMU.
