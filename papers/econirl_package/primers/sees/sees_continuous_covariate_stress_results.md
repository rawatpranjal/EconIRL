# SEES Continuous-Covariate Stress Probe

This probe uses product-grid states encoded as continuous covariates. It tests the current package's encoded-state SEES path and reports the NFXP state-grid explosion.

Scope: this is not yet a true sparse/collocation continuous-state SEES implementation. The current package still materializes dense transition tensors, so very large grids are skipped rather than attempted.

## Grid Explosion

| State dim | States | NFXP value unknowns | SEES value unknowns | Dense transition GiB | NFXP status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5 | 5 | 5 | 0.000 | run |
| 2 | 25 | 25 | 25 | 0.000 | run |
| 3 | 125 | 125 | 40 | 0.000 | run |
| 4 | 625 | 625 | 40 | 0.009 | skip |
| 5 | 3125 | 3125 | 40 | 0.218 | skip |
| 6 | 15625 | 15625 | 40 | 5.457 | skip |
| 7 | 78125 | 78125 | 40 | 136.424 | skip |
| 8 | 390625 | 390625 | 40 | 3410.605 | skip |

## Actual Runs

| Dim | States | Method | Time sec | Converged | Param RMSE | Policy TV | Value RMSE | Bellman | Notes |
| ---: | ---: | --- | ---: | :---: | ---: | ---: | ---: | ---: | --- |
| 2 | 25 | SEES | 6.114 | no | 0.253368 | 0.008223 | 0.331048 | 6.722e-04 | encoded_state |
| 2 | 25 | NFXP | 4.828 | yes | 2.109973 | 0.007317 | 0.273437 |  |  |
| 3 | 125 | SEES | 7.213 | no | 0.272149 | 0.069052 | 0.744243 | 1.539e-02 | encoded_state |
| 3 | 125 | NFXP | 5.734 | yes | 0.119515 | 0.019631 | 0.157854 |  |  |

## Interpretation

- NFXP's exact value object has one unknown per grid state and must solve a full Bellman problem at each theta.
- SEES uses a lower-dimensional encoded-state value basis in the optimization, but the current implementation still pays dense-transition costs.
- The paper-level continuous-state advantage requires a sparse or simulation/collocation SEES path that evaluates expectations without a full S by S transition tensor.
