# CCP/NPL Estimator Design

## Goal

Add CCP-based estimators (Hotz-Miller and NPL) to econirl, benchmarked against NFXP on Rust bus data.

## Background

The CCP (Conditional Choice Probability) approach avoids solving the full dynamic programming problem by using observed choice frequencies to invert the value function. Key references:

- Hotz & Miller (1993): Original CCP inversion theorem
- Aguirregabiria & Mira (2002, 2010): NPL algorithm and survey

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CCP variant | Both Hotz-Miller (K=1) and NPL (K>1) | NPL is iterated Hotz-Miller; unified via `num_policy_iterations` param |
| Initial CCP estimation | Frequency estimator | Simple, consistent, matches literature |
| Value function recovery | Matrix inversion | Closed-form, standard approach (eq 42 in A&M 2010) |

## Algorithm

### Step 1: Estimate Initial CCPs from Data

```python
P̂(a|s) = N(s,a) / N(s)  # frequency estimator
```

Add small epsilon to avoid log(0) for unseen state-actions.

### Step 2: Hotz-Miller Inversion (repeated K times for NPL)

For each policy iteration k = 1, ..., K:

**a) Compute emax correction for logit errors:**
```
e(a,x) = γ - log(P(a|x))
```
where γ ≈ 0.5772 is Euler's constant.

**b) Compute valuation matrix W^P (equation 42 from A&M 2010):**
```
F_π = Σ_a P(a) ⊙ F_x(a)           # policy-weighted transitions
W^P = (I - β·F_π)⁻¹ · Σ_a P(a) ⊙ [z(a), e(a)]
```
where:
- `P(a)` is column vector of CCPs {P(a|x) : x ∈ X}
- `z(a)` is column vector of flow utilities {u(a,x) : x ∈ X}
- `e(a)` is column vector of emax corrections {e(a,x) : x ∈ X}
- `⊙` is element-wise multiplication (broadcasting)
- `F_x(a)` is transition matrix for action a

**c) Maximize pseudo-likelihood:**
```
θ̂_K = argmax Σ_{i,t} log P(a_it | x_it; θ, W^P)
```

**d) Update CCPs (for NPL, k > 1):**
```
P̂_K(a|x) = exp(v(a,x; θ̂_K)) / Σ_j exp(v(j,x; θ̂_K))
```

### Step 3: Return EstimationResult

Return final parameters, value function, policy, standard errors.

## Key Insight from Literature

"All the estimators in this sequence are asymptotically equivalent to partial MLE" (A&M 2010, below eq 44).

This means:
- K=1 (Hotz-Miller) has same asymptotic variance as NFXP/MLE
- NPL iterations reduce finite-sample bias but don't improve asymptotic efficiency
- Upon convergence, NPL gives exact MLE

## File Structure

```
src/econirl/estimation/ccp.py      # CCPEstimator class
tests/test_ccp_estimator.py        # Unit tests
tests/benchmarks/bench_ccp_nfxp.py # Benchmark comparison
```

## CCPEstimator Class

```python
class CCPEstimator(BaseEstimator):
    def __init__(
        self,
        num_policy_iterations: int = 1,  # K=1: Hotz-Miller, K>1: NPL
        ccp_min_count: int = 1,
        convergence_tol: float = 1e-6,   # For NPL convergence check
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
    ): ...

    @property
    def name(self) -> str:
        if self._num_policy_iterations == 1:
            return "Hotz-Miller (CCP)"
        return f"NPL (K={self._num_policy_iterations})"

    def _estimate_ccps_from_data(self, panel, num_states, num_actions) -> Tensor:
        """Frequency estimator: P̂(a|s) = N(s,a) / N(s)"""

    def _compute_emax_correction(self, ccps) -> Tensor:
        """e(a,x) = γ - log(P(a|x)) for logit errors"""

    def _compute_valuation_matrix(self, ccps, transitions, beta) -> Tensor:
        """W^P = (I - β·F_π)⁻¹ · Σ_a P(a)⊙[u(a), e(a)]"""

    def _optimize(self, panel, utility, problem, transitions, ...) -> EstimationResult:
        """Main CCP/NPL algorithm"""
```

## Benchmark Comparisons

Compare against NFXP on Rust bus data:

1. **Parameter recovery** - Estimates within tolerance of NFXP
2. **Standard errors** - Relative difference in asymptotic SEs
3. **Speed** - Wall-clock time ratios
4. **Convergence** - NPL iterations to match NFXP estimates

### Expected Output

```
================================================================================
                    CCP vs NFXP Benchmark on Rust Bus Data
================================================================================
Environment: RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)
Data: 500 individuals × 100 periods = 50,000 observations

                          NFXP          Hotz-Miller      NPL (K=10)
--------------------------------------------------------------------------------
Parameters:
  operating_cost        0.00102          0.00098          0.00101
  replacement_cost      3.012            2.987            3.009

Standard Errors:
  operating_cost        0.00008          0.00009          0.00008
  replacement_cost      0.045            0.051            0.046

Estimation Time:         12.3s            0.8s             2.1s
Speedup vs NFXP:         1.0x            15.4x             5.9x

Log-Likelihood:         -18234.5        -18236.1         -18234.6
Prediction Accuracy:     78.2%           77.9%            78.1%

NPL Convergence:          N/A             N/A           6 iterations
--------------------------------------------------------------------------------
```

## References

- Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities and the Estimation of Dynamic Models." *Review of Economic Studies*, 60(3), 497-529.
- Aguirregabiria, V. and Mira, P. (2002). "Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models." *Econometrica*, 70(4), 1519-1543.
- Aguirregabiria, V. and Mira, P. (2010). "Dynamic discrete choice structural models: A survey." *Journal of Econometrics*, 156(1), 38-67.
