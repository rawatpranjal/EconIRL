# SEES Linear Reward, Value-Spline Benchmark

The reward is action-specific linear in state progress. SEES approximates only the value function with a cubic B-spline basis.

| Seed | Start | Param RMSE | Policy TV | Value RMSE | Q RMSE | Bellman | Grad | Opt flag | Pass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 43 | near_truth | 0.057953 | 0.011652 | 0.036791 | 0.040670 | 2.656e-03 | 2.406e-04 | no | yes |
| 43 | zero | 0.057953 | 0.011652 | 0.036791 | 0.040670 | 2.656e-03 | 2.406e-04 | no | yes |
| 43 | random | 0.057953 | 0.011652 | 0.036791 | 0.040670 | 2.656e-03 | 2.406e-04 | no | yes |
| 65 | near_truth | 0.011144 | 0.008017 | 0.033719 | 0.035461 | 2.818e-03 | 1.240e-03 | no | yes |
| 65 | zero | 0.011144 | 0.008017 | 0.033719 | 0.035461 | 2.818e-03 | 1.240e-03 | no | yes |
| 65 | random | 0.011144 | 0.008017 | 0.033719 | 0.035461 | 2.818e-03 | 1.240e-03 | no | yes |
| 91 | near_truth | 0.041280 | 0.011033 | 0.043113 | 0.046469 | 2.580e-03 | 2.556e-03 | no | yes |
| 91 | zero | 0.025072 | 0.011995 | 0.046803 | 0.051095 | 2.717e-03 | 5.420e-04 | no | yes |
| 91 | random | 0.025072 | 0.011995 | 0.046803 | 0.051095 | 2.717e-03 | 5.420e-04 | no | yes |

Summary: 9/9 runs passed; worst parameter RMSE 0.057953; worst policy TV 0.011995; worst value RMSE 0.046803; worst Q RMSE 0.051095.

Value basis compression: 20/31. The optimizer flag is the strict solver gradient flag and is reported separately from recovery gates.
