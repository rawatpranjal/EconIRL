# Max Margin Planning

| Category | Citation | Reward | Transitions | SEs | Scales | Transfer |
|----------|----------|--------|-------------|-----|--------|----------|
| Inverse | Ratliff, Bagnell, and Zinkevich (2006) | Linear | Yes | Bootstrap only | Yes | No |

## Background

Max Margin IRL (Abbeel and Ng 2004) learns reward weights by finding a reward function that makes the expert's policy better than any other policy by a maximum margin. The original algorithm formulates this as a quadratic program (QP) with constraint generation: at each iteration, find the most violating policy, add a constraint, and re-solve the QP. This works for small problems but the QP grows with each iteration and becomes expensive for large state spaces.

Ratliff, Bagnell, and Zinkevich (2006) reformulated the problem using subgradient descent with loss-augmented inference. Instead of accumulating constraints in a growing QP, each iteration solves one MDP with a modified reward that includes a task-specific loss function. The subgradient of the structured hinge loss is simply the feature difference between the loss-augmented policy and the expert. This makes MMP as scalable as the underlying MDP solver.

## Key Equations

The MMP objective minimizes the regularized structured hinge loss.

$$
\min_\theta \frac{\lambda}{2} \|\theta\|^2 + \frac{1}{N} \sum_i \max_\pi \left[ \Delta(\pi^*, \pi) + \theta^\top (\phi(\pi) - \phi(\pi^*)) \right]
$$

The inner maximization is loss-augmented inference, which solves the MDP with reward $R_{\text{aug}}(s,a) = \theta^\top \phi(s,a) + \Delta(s,a)$.

## Pseudocode

```
MMP(D_expert, p, beta, sigma, max_iter):
  1. Initialize theta = 0
  2. Compute expert features mu_star
  3. For each iteration t:
     a. R_aug(s,a) = theta^T phi(s,a) + scale * Delta(s,a)
     b. Solve MDP under R_aug to get most-violating policy
     c. Compute features mu_hat of most-violating policy
     d. Subgradient: g = lambda * theta + (mu_hat - mu_star)
     e. Update: theta = theta - eta_t * g
  4. Return theta
```

## Strengths and Limitations

MMP is the scalable version of margin-based IRL. The computational cost per iteration is one MDP solve plus one feature expectation computation, the same cost per iteration as MCE-IRL. Unlike the original Max Margin IRL, MMP does not accumulate constraints and its per-iteration cost stays constant. The loss-augmented inference step also allows incorporating task-specific loss functions that penalize specific types of policy errors.

The limitation relative to MCE-IRL is the lack of a probabilistic model. MMP produces a point estimate with no natural likelihood, so standard errors require bootstrap. The margin-based objective can also be sensitive to the choice of loss function and regularization strength.

## In econirl

The implementation is `MaxMarginPlanningEstimator` in `econirl.contrib.max_margin_planning`. Configuration is via `MMPConfig`, which controls learning rate (default 0.1), schedule ("1/sqrt(t)" by default), regularization lambda (default 0.01), loss type ("policy_kl" or "trajectory_hamming"), and inner solver parameters.

## References

- Ratliff, N., Bagnell, J.A., & Zinkevich, M. (2006). Maximum Margin Planning. *ICML 2006*.
- Abbeel, P. & Ng, A.Y. (2004). Apprenticeship Learning via Inverse Reinforcement Learning. *ICML 2004*.
