---
orphan: true
---

# Estimators

EconIRL is a research build. It centers on one reference estimator, the nested
fixed point (NFXP), and treats the rest as answers to specific complications that
break NFXP's canonical case.

Read this page as the menu, not as the evidence. The detailed assumptions,
identification arguments, examples, and validation results live on the estimator
pages linked below.

The estimators are organized into two sections:

- [Core Estimators](estimators/core.md) - NFXP, the exact reference replicated to
  Rust (1987) Table IX.
- [Other Estimators](estimators/other.md) - every other estimator, grouped by the
  source of complexity it answers.

For the reasoning behind the split, the canonical NFXP case, and side-by-side
decision tables, see
[Choosing and Comparing Estimators](comparing_estimators.md).
