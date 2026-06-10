# SEES Rust Random-Start Benchmark

Finite-sample recovery is measured against NFXP on the same simulated panel.

| Mode | Passes | Max param RMSE | Max policy TV | Max value RMSE | Max Q RMSE | Max Bellman | Max grad norm | Optimizer flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 9/9 | 0.000859 | 0.000046 | 0.005762 | 0.005820 | 3.365e-04 | 1.062e-03 | 0/9 |
| q | 9/9 | 0.001960 | 0.000109 | 0.015739 | 0.015420 | 9.297e-04 | 8.793e-04 | 0/9 |
| ev | 9/9 | 0.002212 | 0.000082 | 0.014964 | 0.014944 | 8.389e-04 | 6.783e-04 | 0/9 |
| policy | 9/9 | 0.006692 | 0.000200 | 0.031827 | 0.033279 | 7.661e-04 | 5.564e-03 | 0/9 |
| collocation | 9/9 | 0.000859 | 0.000046 | 0.005762 | 0.005820 | 3.365e-04 | 1.062e-03 | 0/9 |

The optimizer flag is the strict JAXopt gradient flag; recovery gates use the metrics above.
