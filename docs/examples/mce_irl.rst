MCE IRL and Max Margin on Rust Bus
===================================

These examples demonstrate Maximum Causal Entropy IRL and Max Margin IRL on simulated Rust bus engine data. The MCE IRL scripts show feature matching with soft Bellman iteration following Ziebart (2010), recovering structural parameters from expert demonstrations. The Max Margin script shows the alternative approach of finding rewards that maximize the margin between the expert policy and all other policies following Ratliff et al. (2006).

Scripts in this example directory:

- ``mce_irl_bus_demo.py`` -- quick MCE IRL demonstration on a small Rust bus environment
- ``mce_irl_bus_example.py`` -- full MCE IRL estimation with parameter comparison against NFXP
- ``mce_irl_bus_example.ipynb`` -- notebook version with inline plots and diagnostics
- ``max_margin_bus_example.py`` -- Max Margin IRL estimation and comparison on the same Rust bus data
