# Serialized-Content Example

## Important Links

- [AIRL2](../airl2.md)
- [Quick Start](quick_start.md)
- [Pre-Estimation Checks](pre_estimation.md)
- [Simulation Study](validation.md)

Read this page as the concrete serialized-content environment used in the
AIRL2 simulation study. The example shows why exit and absorbing-state anchors
are part of the model, not just implementation details.

The primary validation environment for AIRL2 is a synthetic
serialized-content dynamic choice problem. Individuals decide each period
whether to read (pay and advance), wait (hold position at no cost), or exit
(leave the series permanently). The exit action absorbs the individual into a
terminal state. Two latent segments differ in their sensitivity to quality
versus price.

The self-contained workflow, including data construction and fitting, is
available in
[`airl2_showcase.py`](https://github.com/rawatpranjal/EconIRL/blob/main/examples/serialized-content/airl2_showcase.py).
The public wrapper workflow is also available in the
[`AIRL2 applied notebook`](https://github.com/rawatpranjal/EconIRL/blob/main/examples/airl2/airl2_applied_workflow.ipynb).

## Interpreting Segment Outputs

Estimated segment labels can appear in either order relative to the
data-generating labels. The simulation study runner finds the minimum-cost permutation
between estimated and true segments before reporting accuracy and NRMSE
metrics. When ground-truth rewards are available, choose one permutation that
minimizes total normalized reward error across segments before computing
per-segment metrics.

## Replication Boundary

This page describes the serialized-content environment used by the package
examples. It is not a full replication of the LSW (2026) empirical results.
The simulation measures recovery on one synthetic cell whose data-generating
process is fully specified; see the
[Simulation Study](validation.md) page. The empirical serialized-content
analysis requires proprietary consumption data and is outside this package
repository.
