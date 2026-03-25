# Rust (1987) Bus Engine Replacement — Replication Notes

## Paper

Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher." *Econometrica*, 55(5), 999-1033.

## The Problem

Harold Zurcher, superintendent of the Madison Metropolitan Bus Company, observes bus mileage each month and decides whether to keep the current engine or replace it. Replacement has a fixed cost RC; operating costs increase with mileage. The goal is to estimate RC and the operating cost parameter from observed choices.

## Data

- **Source**: Original NFXP data files (g870.asc, rt50.asc, t8h203.asc, a530875.asc)
- **162 buses** across 8 groups (we have Groups 1-4 = 104 buses, 8,260 obs)
- **Group 1**: Grumman 870 (15 buses, acquired 1983) — newer, low mileage
- **Group 2**: Chance RT-50 (4 buses, acquired 1981)
- **Group 3**: GMC T8H203 (48 buses, acquired 1979) — 27 replacements
- **Group 4**: GMC A5308 (37 buses, acquired 1975) — 33 replacements, highest mileage
- Groups 1 & 2 have **zero replacements** (newer buses, shorter observation window)
- Rust pooled Groups 1,2,3 together and estimated Group 4 separately (p.1018)

## Model

- **State**: Mileage discretized into 90 bins of 5,000 miles each
- **Actions**: Keep (0) or Replace (1)
- **Cost function** (Model 11, linear): `c(x, theta_1) = 0.001 * theta_11 * x`
- **Discount factor**: beta = 0.9999
- **Shocks**: i.i.d. Type I Extreme Value (logit choice probabilities)
- **Transitions**: Mileage increments {0, 1, 2} bins with probabilities (theta_30, theta_31, theta_32)

## Estimation

Three-stage procedure:
1. **First stage**: Estimate transition probabilities theta_3 from mileage increment frequencies
2. **Second stage**: Estimate structural parameters (RC, theta_11) via NFXP (partial likelihood)
3. **Third stage** (optional): Joint MLE of all parameters (full likelihood)

Our `replicate.py` implements stages 1 and 2.

### Algorithm: NFXP with SA-to-NK Polyalgorithm

Following Iskhakov, Jorgensen, Rust & Schjerning (2016):
- **Inner loop**: Solve Bellman equation via policy iteration (9 iterations at beta=0.9999)
- **Outer loop**: BHHH optimizer with analytical gradient via implicit differentiation
- **Precision**: float64 for inner solver (condition number of (I - beta*P) ~ 10^5)
- **Convergence**: 48 BHHH iterations in 0.3 seconds

The analytical gradient uses the implicit function theorem:
```
(I - beta * P_pi) * dV/dtheta = sum_a pi(a|s) * dphi(s,a)/dtheta
```
One n-by-n linear solve per parameter — no numerical finite differences needed.

## Results vs Paper (Table IX)

| Config | Our RC | Paper RC | Our c | Paper c |
|--------|--------|----------|-------|---------|
| Groups 1,2,3 | 11.99 | 11.73 | 4.93 | 4.83 |
| Group 4 | 10.14 | 10.07 | 2.26 | 2.29 |
| Groups 1,2,3,4 | 9.81 | 9.76 | 2.60 | 2.63 |

All within 1-2% of paper values. Residual differences due to:
- Data processing: 8,260 obs (ours) vs 8,156 (paper)
- Likelihood: partial (choices only) vs full (choices + transitions)

### Transition Probabilities vs Paper (Table V)

| Config | Our theta_30 | Paper | Our theta_31 | Paper |
|--------|-------------|-------|-------------|-------|
| Group 4 | 0.394 | 0.392 | 0.593 | 0.595 |
| All pooled | 0.350 | 0.349 | 0.639 | 0.639 |

### Cross-Validation

- **NFXP vs NPL**: Log-likelihood difference |dLL| = 0.002 (effectively identical)
- **econirl vs ruspy**: Log-likelihood matches to 6 decimal places (0.000000 difference)

## Parameterization

econirl uses a slightly different parameterization than the paper:

| | econirl | Rust (1987) | Conversion |
|---|---------|-------------|------------|
| Operating cost | `operating_cost` | `theta_11` | `theta_11 = operating_cost / 0.001` |
| Replacement cost | `replacement_cost` | `RC` | Same |
| Cost function | `-operating_cost * x` | `-0.001 * theta_11 * x` | Equivalent |

Our `operating_cost = 0.0026` corresponds to Rust's `theta_11 = 2.6`.

## Running

```bash
python examples/rust-bus-engine/replicate.py
```

Output includes: data summary, transition estimates, NFXP results with standard errors, NPL cross-validation, per-group estimation matching Table IX, and ruspy cross-validation.

## References

- Rust, J. (1987). "Optimal Replacement of GMC Bus Engines." *Econometrica* 55(5).
- Rust, J. (2000). "NFXP Manual, Version 6." editorialexpress.com/jrust/nfxp.pdf
- Iskhakov et al. (2016). "Comment on Constrained Optimization Approaches." *Econometrica* 84(1).
- Aguirregabiria, V. & Mira, P. (2002). "Swapping the Nested Fixed Point Algorithm." *Econometrica* 70(4).
- OpenSourceEconomics/ruspy — Python reference implementation of NFXP
