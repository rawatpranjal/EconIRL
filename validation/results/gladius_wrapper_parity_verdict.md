# NeuralGLADIUS unweighted-NLL parity fix — verdict

**PASS.** Fresh adversarial re-check found no cheating: the fix is the unweighted NLL default, the truth is used only to score (never to fit), the oracle is not softened, and the same shipped default generalizes to a second DGP (fleet improved 0.964 -> 0.997).

## Check-by-check

1. **Metric correctness — OK.** `theta_cosine = cosine(coef_, TRUE theta)` (harness.py:184, 251), `scale_ratio = ‖hat‖/‖true‖` (harness.py:46-48, 185), `policy_tv` against `_compute_optimal_policy(env)` (harness.py:152, 179). Truth is consumed only to score recovery; legitimate evaluation.

2. **No target leakage in the fit — OK.** `fit_wrapper` passes only the panel and `features=phi` to `m.fit` (harness.py:169), plus the anchor action and `anchor_rewards = R_true[:, anchor_a]` — one action's reward, the paper's Assumption 3 (harness.py:163-165; consumed at neural_gladius.py:526-539 as the anchor-action Bellman term). It does NOT pass `true_theta`, the optimal policy, or non-anchor rewards into the fit; those are computed after the fit for scoring only (harness.py:150-152, 182-184).

3. **Oracle not softened — OK.** Ablation oracle is `median cosine>=0.95 AND scale in [0.80,1.25] AND TV<=0.04` over seeds [0,1,2] (run_ablation.py:3, 14, 54). Generalization reuses the identical ss-spine gate (run_generalization.py:40-42) and fleet `cosine>=0.90` (run_generalization.py:56). Not relaxed.

4. **The fix is the unweighted NLL, not the init — OK.** Shipped default is unweighted: `class_weights = jnp.ones(...)` unless `_ablate={"class_weighting": True}` (neural_gladius.py:484-489), applied at 522 and 556. Q*-init lives only behind `_ablate.get("q_init_bias")` (neural_gladius.py:292-293), off by default. The ablation confirms init-alone FAILS (median cosine -0.244, results_ablation.json:152) while drop-weighting alone PASSES (0.999, results_ablation.json:157-162). Paper-API `GLADIUSEstimator` NLL is a plain `.mean()` with no weights (gladius.py:630, 693, 763), so unweighted is genuine parity, not an invented target.

5. **Generalization (anti-overfit) — OK.** results_generalization.json (no `_ablate`): ss-spine seeds [0.993, 0.999, 0.999], oracle True; fleet cosine 0.997 with theta [3.15, 1.26, 0.37] vs true [3, 1, 0.5], oracle True. Fleet is a different reward and it improved (0.964 -> 0.997), so the fix is not ss-spine-tuned.

6. **No regression — OK.** Pre-run fast suite 70 passed; slow `TestAnchorScaleRecovery` 1 passed. The test guards the fix: threshold tightened `cos>=0.9 -> 0.93` and docstring now states the class-weighted default collapses cosine to ~0 here (test diff in f8a172c5; tests/test_neural_gladius.py:602-645), so re-introducing the weighting would fail it.

## Concerns

None material. Minor: the parity test gate (cos>=0.93) is looser than the ablation/generalization oracle (cos>=0.95), but both are cleared by the actual results (0.99x), so this is a conservative regression guard, not a softened pass. The truth-uses-for-scoring pattern is consistent across all three harness paths.
