# NFXP vs CCP vs TD-CCP on Large State Spaces

| | |
|---|---|
| **Estimators** | NFXP (SA-then-NK), CCP (Hotz-Miller NPL), TD-CCP (neural AVI) |
| **Environment** | Rust bus engine, 200 mileage bins, beta 0.9999 |
| **Key finding** | CCP is 8 times faster than NFXP via matrix inversion. TD-CCP adds per-feature EV diagnostics at the cost of neural training time. |

## Background

All three estimators solve the same structural estimation problem but handle the Bellman fixed point differently. NFXP (Rust 1987) nests a full Bellman solve inside each optimizer step. When the state space is large and the discount factor is high, the contraction mapping converges slowly and the inner loop dominates runtime. CCP (Hotz and Miller 1993) avoids the inner loop entirely by inverting the Hotz-Miller mapping analytically. This requires known transition matrices but eliminates the inner fixed-point solve. TD-CCP (Adusumilli and Eckardt 2025) replaces the inner loop with neural approximate value iteration. It learns separate neural networks for each utility feature component of the expected value, providing a per-feature decomposition that is useful for diagnostics.

## Setup

The data generating process uses a Rust bus engine with 200 mileage bins, a discount factor of 0.9999, and true parameters theta_c equal to 0.001 and RC equal to 3.0. The data consists of 200 buses observed over 100 periods, yielding 20,000 observations. NFXP uses the hybrid inner solver. CCP runs 5 NPL policy iterations. TD-CCP uses a two-layer MLP with 32 hidden units and 15 approximate value iteration rounds.

## Code

The full script is at ``examples/rust-bus-engine/tdccp_vs_nfxp.py``.

```python
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig

nfxp = NFXPEstimator(inner_solver="hybrid", inner_max_iter=300000)

ccp = CCPEstimator(num_policy_iterations=5)

tdccp = TDCCPEstimator(config=TDCCPConfig(
    hidden_dim=32, avi_iterations=15, n_policy_iterations=3,
))
```

## Results

Single replication with true parameters theta_c equal to 0.001 and RC equal to 3.0.

| Metric | NFXP | CCP | TD-CCP |
|---|---|---|---|
| theta_c | 0.001468 | 0.001467 | 0.000197 |
| RC | 3.0731 | 3.0731 | 3.0557 |
| Bias (theta_c) | 0.000468 | 0.000467 | -0.000803 |
| Bias (RC) | 0.0731 | 0.0731 | 0.0557 |
| Log-likelihood | -4193.14 | -4193.14 | -4192.90 |
| Wall time (seconds) | 160.3 | 20.5 | 359.3 |

## Discussion

NFXP and CCP converge to identical parameters because CCP with NPL iterations converges to the MLE. CCP is 8 times faster because it replaces the costly inner Bellman solve with a matrix inversion that scales as the cube of the state space size. For 200 states this is fast, but for thousands of states the matrix inversion itself becomes expensive.

TD-CCP recovers the replacement cost within 2 percent of the true value but shows more bias on the operating cost. This reflects the neural approximation error in the per-feature EV decomposition. The operating cost is small (0.001) relative to the replacement cost (3.0), making it harder for the neural network to resolve the fine-grained mileage gradient. TD-CCP is slower than both alternatives on this problem because the neural training overhead exceeds the savings from avoiding the inner loop. The advantage of TD-CCP emerges on continuous-state problems where neither NFXP nor CCP can operate without discretization bias.

The per-feature EV decomposition in TD-CCP is a unique diagnostic. Instead of a single opaque value function, the researcher sees how each utility component contributes to the expected continuation value. This can reveal which features are well-identified by the data and which are poorly pinned down.
