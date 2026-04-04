DDC and IRL Equivalence
=======================

These examples explore the theoretical equivalence between dynamic discrete choice estimation and inverse reinforcement learning. The comparison script runs NFXP and MCE-IRL side by side on the same Rust bus data and shows they recover equivalent parameters when features are action-dependent. The equivalence notebook formalizes the mapping between the DDC likelihood and the MaxEnt IRL objective. The GCL notebook extends the comparison to Guided Cost Learning, which uses a neural reward network with adversarial training.

Scripts in this example directory:

- ``ddc_maxent_equivalence.py`` -- formalizes the theoretical mapping between DDC log-likelihood and MaxEnt IRL feature matching
- ``nfxp_vs_mceirl_comparison.ipynb`` -- head-to-head NFXP versus MCE-IRL parameter recovery on simulated Rust bus data
- ``gcl_comparison.ipynb`` -- extends the equivalence analysis to Guided Cost Learning with neural rewards
