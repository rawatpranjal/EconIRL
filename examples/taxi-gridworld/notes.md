# Taxi-Gridworld MCE IRL — Notes

## Problem Setup
- 10x10 grid, 100 states, 5 actions (Left/Right/Up/Down/Stay)
- Deterministic transitions (walls = stay in place)
- Terminal absorbing state at (9,9)
- True reward: R(s,a) = -0.1·step + 10·terminal + 0.1·(-dist/20)

## Experiments

### R(s,a) In-Sample
Both MCE IRL and NFXP converge but with different parameter magnitudes due to scale non-identification. MCE IRL finds small params (-0.06, 0.06, 0.03), NFXP finds huge params (81, 77, 48). The RATIOS differ too — this is because the gridworld with deterministic transitions has very weak identification for absolute parameter values.

### R(s,a) Transfer (deterministic → 10% stochastic)
- NFXP: 82.5% optimal — transfers well despite crazy param magnitudes
- MCE IRL: 7.1% optimal — poor transfer

NFXP transfers better because its huge parameters make the policy very peaked (essentially deterministic), which happens to be near-optimal. MCE IRL's small parameters give a softer policy that degrades under perturbation.

### R(s) State-Only Features
- MCE IRL recovers step_penalty=-0.67, distance_weight=2.40 (correct signs, LL=-28,815)
- MaxEnt IRL recovers step_penalty=-94.6, distance_weight=49.9 (correct signs, LL=-28,553)
- NFXP cannot directly estimate R(s) — the flow utility cancels in P(a|s)

### R(s) Transfer
MCE IRL R(s) → stochastic grid: **54.5% optimal** — much better than R(s,a) transfer (7.1%)!
R(s) is more portable because it doesn't bake in dynamics-specific continuation values.

## Key Insight: Why MCE IRL Can Estimate R(s) But NFXP Can't (Easily)

### NFXP gradient for R(s):
```
dLL/dθ = Σ_t [P(s'|s_t,a_t) - P_π(s'|s_t)]ᵀ · (I - βP_π)⁻¹ · ∂R/∂θ
```
The R(s) flow term cancels in P(a|s), but enters indirectly through V(s') via the Bellman recursion. Identification is **weak** — the signal goes through (I-βP_π)⁻¹ which is ill-conditioned when β→1.

### MCE IRL gradient for R(s):
```
∇L = μ_D - μ_π = empirical state features - model-predicted state features
```
State visitation d_π(s) depends DIRECTLY on R(s) — states with higher R are visited more. This gives a **direct identification channel** that NFXP lacks.

### Summary: Two Identification Channels

| Channel | What it identifies | Strength | Used by |
|---------|-------------------|----------|---------|
| Flow utility | R(s,a) differences | Strong | NFXP, MCE IRL |
| State visitation | R(s) levels | Strong | MCE IRL only |
| Continuation value | R(s) via V(s') | Weak | NFXP only |

**Recommendation**: Use MCE IRL for R(s), NFXP for R(s,a). For transfer, prefer R(s).

## Hacks to Make NFXP Work with R(s)
1. **L2 regularization**: LL(θ) - λ||θ||² prevents scale explosion
2. **Parameter normalization**: project θ → θ/||θ|| after each step
3. **Initialize from MCE IRL**: warm-start NFXP near reasonable R(s) params
4. **Fix one parameter**: anchor one feature weight to pin down scale
5. **CCP inversion**: Hotz-Miller directly recovers V(s) → R(s) from CCPs
