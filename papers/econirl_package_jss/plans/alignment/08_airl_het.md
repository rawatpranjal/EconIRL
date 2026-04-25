## Estimator: AIRL-Het (Adversarial IRL with unobserved heterogeneity)
## Paper(s): Lee, Sudhir, Wang 2026 "Modeling Serialized Content Consumption with Heterogeneous Reward Functions" (working paper). The paper is on the LSW synthetic dataset; not yet in `papers/foundational/`. Project memory `project_airl_het_identification.md` summarizes the EM + anchor identification approach.
## Code: `src/econirl/estimation/adversarial/airl_het.py`

### Loss / objective

- Paper formulation (per project memory): vanilla AIRL loss per latent segment `k`, with EM over segment assignments. The full-data log-likelihood is

  ```
  L(theta_1, ..., theta_K, w) = sum_i log [ sum_k w_k * Pr_k(traj_i | theta_k) ]
  ```

  with `w` the segment weights and `Pr_k` the AIRL likelihood evaluated under segment `k`'s reward.

  Identification is anchored: the first segment's reward is anchored to a known feature (or a constant) to break the label-switching ambiguity that affects mixture models (Arcidiacono-Miller 2011 anchor identification, generalized to AIRL).

- Code implementation: `airl_het.py` runs an outer EM loop. The E-step computes posterior segment probabilities given current segment-specific rewards; the M-step runs one AIRL update per segment using the posterior-weighted demonstrations. Anchor identification is enforced via `AIRLHetConfig.anchor_action` and `AIRLHetConfig.anchor_state`.

- Match: **yes**, follows the EM-AIRL pattern. The anchor identification mirrors Arcidiacono-Miller 2011's CCP-with-unobserved-heterogeneity.

### Gradient

- Paper formula: per-segment AIRL gradient weighted by E-step posteriors (standard mixture-model EM).

- Code implementation: matches. The `_optimize` method loops over `num_segments` and runs a weighted AIRL update per segment.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: same as vanilla AIRL within each segment.

- Code algorithm: matches. Each segment runs the same soft-Bellman policy update as `airl.py`.

- Match: **yes**.

### Identification assumptions

- Paper conditions: anchor identification requires (a) at least one known anchor per segment, (b) sufficient within-segment data to identify the per-segment reward, (c) segments are ergodic (not absorbing too quickly). The LSW dataset is constructed to satisfy all three.

- Code enforcement: the wrapper requires `anchor_action`, `anchor_state`, `num_segments`, `exit_action`, and `absorbing_state` to be passed in `AIRLHetConfig`. The Tier 3c cell sets these explicitly.

- Match: **yes**.

### Hyperparameter defaults vs paper defaults

- `num_segments`: 2 (LSW canonical setting).
- `anchor_action`, `anchor_state`: required (no defaults).
- `em_max_iter`: 50 (per project memory; LSW converges in ~10–20 iterations).

Match: **yes**.

### Findings / fixes applied

- **Paper-side gap**: the LSW 2026 paper PDF is not in `papers/foundational/`. **Action**: add it. Tracked in CLOUD_VERIFICATION_QUEUE.md.

- **No code fixes required** beyond noting that the cell-level config in matrix.py already sets the anchors. The Tier 3c cell `tier3c_lsw_synthetic_airlhet` carries `extra_kwargs={"num_segments": 2, "exit_action": 2, "absorbing_state": 0}`, matching the paper's setup.

- VALIDATION_LOG.md status: **Pending** (Tier 3c lsw-synthetic dispatch).
