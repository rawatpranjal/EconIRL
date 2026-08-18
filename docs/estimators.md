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

- [Core Estimators](estimators/core.md) - the curated main-method roster,
  anchored by NFXP and its exact Rust (1987) Table IX replication.
- [Other Estimators](estimators/other.md) - every implemented estimator outside
  the Core roster.

For the reasoning behind the split, the canonical NFXP case, and side-by-side
decision tables, see
[Choosing and Comparing Estimators](comparing_estimators.md).
