Ziebart MCE IRL Replication
===========================

These examples replicate the original Maximum Causal Entropy IRL results from Ziebart (2010) on a tabular gridworld. The replication script follows the algorithm from the thesis, computing state visitation frequencies via the forward pass and updating reward parameters by gradient ascent on the causal entropy objective. The gridworld runner provides a self-contained environment for testing reward recovery with known ground truth.

Scripts in this example directory:

- ``ziebart_mce_irl_replication.py`` -- replicates Ziebart (2010) MCE IRL on a tabular gridworld with known reward
- ``run_gridworld.py`` -- sets up the gridworld environment and generates expert demonstrations for the replication
