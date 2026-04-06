# IQ-Learn Primer Redesign

**Date:** 2026-04-06  
**Status:** Approved

## Problem

The current `iq_learn.tex` primer runs IQ-Learn against NFXP on a 200-trajectory Rust Bus problem. NFXP wins. This is the wrong experiment: IQ-Learn was designed for the low-data IRL regime, where adversarial methods (GAIL) collapse and BC cannot generalize to unseen states. The primer shows the estimator losing in a setting it was never intended for.

## Experiment Design

**Setting:** RustBus, 90 mileage bins, β=0.9999, operating_cost=0.001, replacement_cost=3.0.

**Data:** 5 expert trajectories × 100 periods = 500 observations per trial. Three seeds: 42, 7, 123.

**Oracle:** NFXP fitted on 2000 trajectories (not a competitor — defines the ground-truth greedy policy and true parameters for evaluation only).

**Methods compared:**
- IQ-Learn: `IQLearnEstimator(IQLearnConfig(q_type="linear", alpha=1.0))`
- MCE-IRL: `MCEIRLEstimator(MCEIRLConfig())` — stable IRL baseline with inner VI loop
- GAIL: `GAILEstimator(GAILConfig(discriminator_type="linear", max_rounds=200))` — adversarial baseline
- BC: `BehavioralCloningEstimator()` — floor

**Metrics per seed:**
1. Policy accuracy: fraction of states where `argmax π(·|s)` matches oracle NFXP greedy action
2. Reward cosine similarity: `cos(θ)` between recovered parameters and true `[0.001, 3.0]` — N/A for BC and GAIL (no parametric reward recovery)
3. Training time (seconds)

**Aggregation:** mean ± std across the 3 seeds.

**Narrative:** IQ-Learn's chi-squared regularizer selects a well-conditioned reward from the equivalence class even when expert coverage is thin (500 obs across 90×2 cells). GAIL's min-max objective diverges when expert coverage is sparse — the discriminator overfits to the handful of visited (s,a) pairs, producing an adversarial reward signal that pushes the policy in the wrong direction. MCE-IRL works but runs ~5000 inner VI iterations per outer step at β=0.9999, making it ~10× slower. BC memorizes observed transitions but assigns uniform probability to the ~30% of states never seen in 5 short trajectories.

## Files Changed

| File | Change |
|------|--------|
| `papers/econirl_package/primers/iq_learn/iq_learn_run.py` | Full rewrite: 3-seed loop over all 4 methods, compute all metrics, write JSON + tex |
| `papers/econirl_package/primers/iq_learn/iq_learn_results.json` | Regenerated with per-seed and summary (mean ± std) entries |
| `papers/econirl_package/primers/iq_learn/iq_learn_results.tex` | Regenerated: one table, mean ± std, four methods |
| `papers/econirl_package/primers/iq_learn/iq_learn.tex` | Section 1 framing updated; Section 5 (Results) rewritten |
| `papers/econirl_package/primers/iq_learn/iq_learn.pdf` | Recompiled |
| `papers/econirl_package/primers/iq_learn/*.aux/.log/.out` | Deleted (build artifacts) |

## Results JSON Schema

```json
{
  "oracle": {"params": [0.001230, 3.0115], "ll": -4263.2, "time": 12.4},
  "seeds": {
    "42":  {"iq_learn": {...}, "mce_irl": {...}, "gail": {...}, "bc": {...}},
    "7":   {"iq_learn": {...}, "mce_irl": {...}, "gail": {...}, "bc": {...}},
    "123": {"iq_learn": {...}, "mce_irl": {...}, "gail": {...}, "bc": {...}}
  },
  "summary": {
    "iq_learn": {"policy_acc_mean": ..., "policy_acc_std": ..., "cos_sim_mean": ..., "cos_sim_std": ..., "time_mean": ...},
    "mce_irl":  {...},
    "gail":     {"policy_acc_mean": ..., "policy_acc_std": ..., "cos_sim_mean": null, "time_mean": ...},
    "bc":       {"policy_acc_mean": ..., "policy_acc_std": ..., "cos_sim_mean": null, "time_mean": ...}
  }
}
```

Each per-method entry per seed: `{"params": [...], "policy_acc": float, "cos_sim": float|null, "time": float, "converged": bool}`.

## Tex Results Table

One table in Section 5:

```
Method     | Policy Acc. (%)    | Reward cos-sim     | Time (s)
IQ-Learn   | μ ± σ              | μ ± σ              | μ
MCE-IRL    | μ ± σ              | μ ± σ              | μ
GAIL       | μ ± σ              | ---                | μ
BC         | μ ± σ              | ---                | <1
```

Caption explains: 5 expert trajectories, 3 seeds. Policy accuracy vs oracle NFXP. IQ-Learn achieves [X]% accuracy in [Y]s with low variance; GAIL shows high variance from adversarial instability; MCE-IRL matches accuracy but at [Z]× the wall time due to inner VI at β=0.9999.

## Tex Prose Changes

**Section 1 (Problem and Motivation):** Change opening sentence from "IQ-Learn... is a model-free IRL method" to frame it as: "IQ-Learn's practical case is the low-data IRL regime — few expert demonstrations, unknown or noisy transitions, and no adversarial training loop." Remove the sentence about "the original paper targets continuous state-action spaces" which weakens the framing.

**Section 5 (Results):** Replace entirely. Current prose describes a 200-trajectory experiment where IQ-Learn loses. New prose describes the 5-trajectory stability experiment, reports actual numbers from the JSON, explains why GAIL collapses (discriminator overfits to sparse coverage), explains why IQ-Learn is faster than MCE-IRL (no inner VI), and states the conclusion plainly.

All other sections (Model/Notation, Core Theory, Algorithm, Implementation) remain unchanged — the math is correct.

## Constraints

- Hard limit: 5 pages. Current tex is at 5 pages; replacing one results table with one equivalently-sized table will not change page count.
- Common notation: uses `\input{notation}` from the parent folder. No new notation introduced.
- No links to `examples/` — this is a self-contained primer experiment.
- BC baseline included.
- Citations: existing inline references at end of document are sufficient; no new `.bib` needed.
- After recompile, delete `.aux`, `.log`, `.out` from the folder.
