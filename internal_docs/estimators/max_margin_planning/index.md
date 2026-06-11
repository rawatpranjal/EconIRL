# Max Margin Planning Internal Notes

Maximum Margin Planning is the Ratliff-Bagnell-Zinkevich structured
large-margin variant. It replaces the QP-style constraint generation of Max
Margin IRL with loss-augmented inference and subgradient descent. The package
keeps it as a contrib method only.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD page: none currently.
- Contrib implementation: `../../../src/econirl/contrib/max_margin_planning.py`.
- Tests: `../../../tests/contrib/test_max_margin_planning_core.py`.
- Validation result: none currently.

## Algorithm

The objective has the structured-SVM form:

```text
min_theta lambda/2 ||theta||^2
  + average_i max_policy [loss(expert_i, policy)
      + theta' (phi(policy) - phi(expert_i))].
```

The package implementation alternates:

1. estimate the empirical expert policy from demonstrations;
2. run loss-augmented value iteration;
3. compute expert and policy feature expectations;
4. take a subgradient step with L2 regularization;
5. repeat until gradient norm or iteration limits stop the run.

Supported loss types include `policy_kl` and `trajectory_hamming`. The inner
solver can use value iteration or hybrid iteration.

## Difference From Max Margin IRL

| Dimension | Max Margin IRL | Max Margin Planning |
| --- | --- | --- |
| Optimization | Constraint generation with QP-like solve | Subgradient descent |
| Main inner step | Most-violating policy under reward | Loss-augmented inference |
| Regularization | Norm/anchor constraint | L2 regularization |
| Package state | Wrapper plus contrib | Contrib only |
| Validation | No tracked JSON | No tracked JSON |

## Current Release State

This estimator has a contrib implementation and tests, but no tracked
validation JSON and no public RTD guide. It should not be cited as release
evidence until a known-truth artifact exists.

## Debugging Order

1. Verify empirical expert policy estimation.
2. Check loss type and loss scale.
3. Confirm loss-augmented value iteration converges.
4. Inspect feature-difference subgradients.
5. Check learning-rate schedule and L2 penalty.
6. Compare final policy only after margin loss stabilizes.

## Implementation Paths

- Contrib estimator: `../../../src/econirl/contrib/max_margin_planning.py`.
- Core tests: `../../../tests/contrib/test_max_margin_planning_core.py`.

## Public Documentation Boundary

Keep Max Margin Planning as an internal contrib page until validation artifacts
and public-use guidance are added.
