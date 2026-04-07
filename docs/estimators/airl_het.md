# AIRL-Het

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Lee, Sudhir, and Wang (2026) | Linear / Tabular | No (adversarial) | No | No |

[**Primer (PDF)**](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/aairl/aairl.pdf)

## What this estimator does

AIRL-Het recovers segment-specific reward functions from mixed-population data where consumers have unobserved heterogeneous preferences. It extends AIRL (Fu et al. 2018) with two structural contributions: anchor constraints that make action-dependent rewards uniquely identified, and an EM algorithm that discovers $K$ latent consumer types, each with its own reward and policy.

Standard IRL methods assume a homogeneous population and recover a single reward function. On platforms with multiple consumer types (quality-lovers vs. price-sensitive, subscribers vs. browsers), this average reward misrepresents every segment and prevents segment-specific counterfactuals. AIRL-Het separates the types.

## How it works

The discriminator logit decomposes into reward and shaping:

$$
f^k(s,a,s') = g_\theta^k(s,a) + \beta\, h_\phi^k(s') - h_\phi^k(s).
$$

Anchor constraints $g_\theta^k(s, a_{\text{exit}}) = 0$ and $h_\phi^k(s_{\text{abs}}) = 0$ uniquely pin down the reward at convergence: $g_\theta^k = r^*_k$ and $h_\phi^k = V^*_k$. Segment membership is inferred via EM: the E-step computes posterior probabilities from trajectory log-likelihoods under each segment's policy; the M-step runs a weighted AIRL inner loop per segment. A within-individual consistency weight encourages the same consumer to map to the same segment across multiple observations.

## Quick start

```python
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig

config = AIRLHetConfig(
    num_segments=2,
    exit_action=2,        # action index with zero reward (required, no default)
    absorbing_state=50,   # terminal state index (required, no default)
    reward_type="linear",
    max_em_iterations=40,
    max_airl_rounds=80,
    verbose=True,
)
estimator = AIRLHetEstimator(config)
summary = estimator.estimate(panel, utility, problem, transitions)

# Access segment-specific outputs
posteriors  = summary.metadata["segment_posteriors"]   # shape (n_trajs, K)
assignments = summary.metadata["segment_assignments"]  # hard segment labels
policies    = summary.metadata["segment_policies"]     # list of K policy matrices
```

Output with `verbose=True`:

```
EM-AIRL:  20%|██       | 8/40 [02:14<08:58, LL=-12043.2, dLL=0.00312, priors=[0.54 0.46]]
```

## When to use it

Use AIRL-Het when the data come from a mixed population and segment-specific counterfactuals matter. The classic application is content consumption where consumers differ in quality sensitivity and price sensitivity. On tabular problems with a known homogeneous population, NFXP is faster and more accurate. The estimator requires explicit `exit_action` and `absorbing_state` indices with no sensible defaults.

## References

- Lee, S., Sudhir, K., and Wang, T. (2026). Modeling Serialized Content Consumption: Adversarial IRL for DDC. Yale SOM working paper.
- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR*.
