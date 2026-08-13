# Simulation Study

## Important Links

- [TD-CCP overview](../tdccp.md)
- [Quick Start](quick_start.md)
- [Rust Bus Engine Example](rust_bus.md)
- [Estimator Replications](../../replications.md)
- [Simulation Studies](../../simulation_studies/index.md)

The evidence separates the published paper comparison from independent package
checks. The paper comparison uses the authors' design and seed schedule. The
other studies test inference, prediction, counterfactuals, nonlinear recursive
terms, and higher-dimensional state encodings.

## Published Table E.1

The paper comparison follows Online Appendix Table E.1 of Adusumilli and
Eckardt (2025). EconIRL fitted 1,000 panels regenerated from the official code
and seed schedule. It reproduced all 12 published means and empirical standard
deviations to at least four significant figures.

| Estimator | Panels | Largest mean gap | Largest empirical SD gap |
| --- | ---: | ---: | ---: |
| Plug-in semigradient | 1,000 | 6.88e-06 | 2.12e-06 |
| Locally robust semigradient | 1,000 | 7.05e-08 | 1.03e-07 |

All locally robust fits met the optimizer criterion. The plug-in optimizer
stopped short on 318 panels. Those fits remained finite and reproduced the
published sampling summaries. This result checks the repeated point
estimates from the paper. The next study evaluates confidence intervals under
a separate larger-sample design.

The complete comparison is
[`tdccp_table_e1.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_table_e1.json).
The runner is
[`paper_table_e1_mc.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/tdccp/paper_table_e1_mc.py).

## Locally Robust Inference

The inference study uses 1,000 independently simulated stationary panels. Each
panel has 3,000 individuals and 30 periods. It evaluates the Algorithm 2
correction and individual-clustered standard errors.

| Parameter | Standardized bias | Mean SE / empirical SD | 95% coverage | Tail miss L / U |
| --- | ---: | ---: | ---: | ---: |
| $\theta_0$ | 0.0064 | 0.9946 | 0.949 | 0.020 / 0.031 |
| $\theta_1$ | 0.0672 | 1.0076 | 0.953 | 0.029 / 0.018 |
| $\theta_2$ | 0.0049 | 0.9882 | 0.945 | 0.027 / 0.028 |

The initial distribution is stationary to numerical precision. This matters
because the backward recursion learns predecessor distributions from observed
trajectories.

The full result is
[`tdccp_inference.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_inference.json).

## Prediction and Counterfactuals

The hard-problem study uses 20 independent panels. Every fit was usable. The
median relative parameter error was 0.0482 and the 90th percentile was 0.0685.
Mean policy total variation was 0.00426. The mean excess negative log
likelihood was 5.24e-05 and the mean excess Brier score was 2.71e-05.

| Intervention | Oracle policy effect | Fitted policy TV | Value regret |
| --- | ---: | ---: | ---: |
| Reward change | 0.1689 | 0.00260 | 0.00109 |
| Transition change | 0.0495 | 0.00435 | 0.00156 |

The oracle policy effect measures whether the intervention changes behavior.
Policy total variation and regret then measure recovery under that changed
environment. See
[`tdccp_ready.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_ready.json)
for the individual runs.

## Bootstrap Intervals

The bootstrap study resamples whole individual trajectories. It uses 50 panels
and 99 resamples per panel. All 4,950 draws succeeded.

| Parameter | Usable panels | 95% coverage | Mean interval width |
| --- | ---: | ---: | ---: |
| Replacement cost | 50 of 50 | 0.94 | 0.1831 |
| Condition cost | 50 of 50 | 0.98 | 0.3165 |

The detailed result is
[`tdccp_bootstrap_calibration.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_bootstrap_calibration.json).

## Neural and High-Dimensional Checks

The approximate value iteration study uses 30 paired stationary panels. All
fits were usable. The largest mean relative parameter error was 0.0624. The
locally robust aggregate bias was 0.0625, compared with 0.1130 for the plug-in
version.

The encoded-state study uses 30 paired seeds. Moving from zero to 20 nuisance
features changed the mean parameter RMSE by a factor of 1.006. With 20 nuisance
features, shuffling successor links increased the mean absolute error of
$\theta_1$ by a factor of 24.7. The shuffled result shows that the estimator
uses the observed dynamics rather than fitting only current choices.

See
[`tdccp_neural_avi.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_neural_avi.json)
and
[`tdccp_highdim.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_highdim.json)
for the complete records.

## Package Checks

A fitted estimator was saved and loaded in a fresh Python process from the
built wheel. Coefficients, expected features, choice probabilities,
counterfactual policies, and counterfactual values matched exactly. The
[`tdccp_serialization.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_serialization.json)
record reports the installed module path, interpreter, commit, and observed
gaps.

The [dedicated notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/tdccp/tdccp_applied_workflow.ipynb)
covers pre-estimation checks, fitting, inference, prediction, simulation,
counterfactuals, and serialization.

## Reproduce the Study

Run the hard-problem study from the repository root. Progress is written to a
temporary log so the terminal reports one stable result.

```bash
PYTHONPATH=src:. uv run python validation/estimators/tdccp/ready.py \
  --output validation/results/tdccp_ready.json >/tmp/tdccp-ready.log
PYTHONPATH=src:. uv run python -c \
  'import json; p=json.load(open("validation/results/tdccp_ready.json")); s=p["summary"]; print("TD-CCP hard-problem study: {} ({} of {} usable)".format(s["status"], s["usable_replications"], s["completed_replications"]))'
```

**Result**

```text
TD-CCP hard-problem study: ready (20 of 20 usable)
```
