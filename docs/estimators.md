---
orphan: true
---

# Estimators

EconIRL is a research build. It centers on one reference estimator, the nested
fixed point (NFXP), and treats the rest as answers to specific complications that
break NFXP's canonical case.

The estimators are organized into two sections:

- [Core Estimators](estimators/core.md) - NFXP, the exact reference replicated to
  Rust (1987) Table IX.
- [Other Estimators](estimators/other.md) - every other estimator, grouped by the
  source of complexity it answers.

For the reasoning behind the split and a side-by-side decision table, see
[Choosing an Estimator](estimators/landscape.md).
