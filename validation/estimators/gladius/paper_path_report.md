# GLADIUS Paper-Path Diagnosis and Repair

Date: 2026-08-18

Status: the small and staged paper cells are repaired, but the release remains
unqualified until the uncapped 6-by-20 Table 2 run passes. No bootstrap, full
sweep, push, publication, or release action was performed during this repair.

## What Failed

The capped ten-seed run failed every Table 2 cell:

| Trajectories | Mean MAPE (%) | Allowed (%) |
|---:|---:|---:|
| 50 | 8.97 | 6.00 |
| 250 | 2.34 | 1.86 |
| 500 | 1.70 | 0.95 |
| 1,000 | 7.56 | 0.96 |
| 2,500 | 11.50 | 0.25 |
| 5,000 | 14.19 | 0.24 |

The result was not evidence that more seeds were needed. The remaining seeds
could not rescue those frozen means. It was evidence that the training
protocol was not comparable across sample sizes.

## Root Causes

1. **The earlier receipt did not describe the network that ran.** The driver
   labeled the estimator `shared_trunk` without setting `network_mode`, so it
   trained the lower-level `separate` default. The repaired receipt now checks
   the effective estimator metadata field by field and fails closed on drift.

2. **The package omitted several author-loop mechanics.** The checked author
   implementation batches whole trajectories, uses a shared Q/zeta trunk,
   Xavier-normal weights, a -55 output bias, summed zeta regression loss,
   unclipped zeta gradients, value-clipped Q gradients, separate AdamW states,
   and epoch-based learning-rate decay. The paper driver now sets and records
   those choices explicitly.

3. **The absolute reward level has a weak gradient.** A uniform Q/zeta shift
   changes the anchor residual by only `(1 - beta)c`; here `beta=0.95`. Choice
   likelihood and action-difference tests can therefore look good while raw
   reward MAPE is wrong. The Table 2 path now projects the common Q/zeta level
   onto the known replacement reward after each Q update. This uses only the
   paper's known anchor, leaves current action differences and policy unchanged,
   and is disclosed as a package repair beyond the author code.

4. **Fixed batch size starved the small cell of Q updates.** N=50 has 40
   training trajectories. With trajectory batch size 32 and alternating zeta/Q
   batches, each epoch contains only two batches and therefore one Q update.
   This produced a heavy-tailed optimizer failure: most seeds eventually fit,
   but a few 23-50% MAPE runs dominated the mean. The protocol now uses batch
   size 2 at N=50 and 5 in larger cells. This gives 10, 20, 40, 80, 200, and
   400 Q updates per epoch as N grows.

5. **The update cap changed the experiment with N.** A global optimizer-update
   cap shortened the number of epochs sharply as N grew. The related author
   configuration decays learning rate by epoch, and the N=500 probe was still
   improving at epoch 192. The final qualification protocol therefore uses 800
   epochs for every cell and no update cap.

6. **The tests checked surfaces before scientific behavior.** Formula, shape,
   determinism, and serialization tests did not require the literal paper path
   to recover cardinal rewards. Cheap one-state and three-state recovery gates
   now run before Table 2. A paper-bus regression also fixes the previously bad
   N=50 seed 3 at 200 epochs and the adaptive batch size.

## Cheap Repair Evidence

No bootstrap was run. These probes use the exact 20-state paper DGP and the
simulation-only true-MAPE checkpoint rule used by the checked author experiment.

| Probe | Fixed batch 32 | Adaptive batch | Bound | Result |
|---|---:|---:|---:|---|
| N=50, seed 3 | 50.37% at 800 epochs | 0.59% at 200 epochs | 6.00% | repaired |
| N=50, seed 6 | 23.20% at 800 epochs | 0.54% at 200 epochs | 6.00% | repaired |
| N=50 aggregate | 8.97% in old ten-seed run | 4.32% for seeds 0-6 at 200 epochs | 6.00% | staged pass |
| N=250, seed 0 | not rerun | 0.48% at 200 epochs | 1.86% | staged pass |
| N=500, seed 0 | not rerun | 0.99% at 200 epochs | 0.95% | still improving |
| N=500, seed 0 | not rerun | 0.89% at 400 epochs | 0.95% | staged pass |

For the seven-seed N=50 check, the NFXP control averaged 3.57% MAPE, close to
the paper's 3.62%. This supports the existing conclusion that the DGP and
metric are correct. The adaptive GLADIUS mean was 4.32%, below the frozen 6.00%
acceptance bound. Individual GLADIUS seeds still ranged from 0.54% to 10.19%,
so the final 20-seed mean remains necessary.

## What Is and Is Not Paper-Exact

The DGP, raw-reward MAPE, 80/20 trajectory split, 2-by-10 Q architecture,
minimax objective, known replacement anchor, trajectory batching, and
simulation-only best-MAPE checkpoint follow the paper or checked author code.
The paper does not publish the Table 2 batch size, and the available author
configuration is for a related high-dimensional experiment rather than the
zero-dummy Table 2 cell.

The adaptive batch rule and anchor-level projection are EconIRL stability
repairs. They must remain visible in the receipt. The result may qualify the
package's paper-scale replication gate, but it must not be described as a
byte-for-byte reproduction of an unpublished Table 2 training configuration.

## How to Improve Further

- Treat Q updates per epoch, full-data passes, and epochs as separate recorded
  quantities. Never compare sample-size cells using only a global update cap.
- Keep the scientific gate hierarchy: one-state cardinal recovery, small-bus
  recovery, bad-seed N=50, staged N=250 and N=500, then the full 6-by-20 run.
- Preserve the raw reward level, anchor error, policy error, and structural
  projection as distinct diagnostics. Policy or cosine agreement cannot stand
  in for absolute reward recovery.
- Keep oracle checkpoint selection inside simulation validation. A public fit
  cannot observe true reward MAPE and needs a non-oracle stopping rule before
  these training choices can be promoted to the public estimator default.
- If the 800-epoch run still has large outliers, investigate the nonsmooth
  author stabilization `mean(abs(td^2 - beta^2 dev^2))` against the signed
  empirical minimax objective. Do not tune this after seeing the full 20-seed
  qualification result; freeze a small diagnostic design first.
- Consider a future paired-batch implementation of paper Algorithm 1, where
  each iteration performs both zeta and Q updates on independently drawn
  batches. That is a separate protocol change and needs its own simple-case and
  bad-seed evidence before use.

## Remaining Release Gate

Run the four five-seed shards at all six sizes with `--max-epochs 800`, without
`--max-updates` or `--batch-size`, then merge them. Only a complete 20-seed
receipt whose six means pass the frozen thresholds can turn the qualification
green. Bootstrap calibration stays deferred until that paper gate passes.
