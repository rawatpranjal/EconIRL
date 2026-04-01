"""Sample size analysis: when does structural estimation beat reduced-form?

Sweeps N (number of individuals) and measures Type II counterfactual
error for each method. The interesting question is: at what sample
size does AIRL+anchors (slow convergence, correct target) beat
reduced-form Q (fast convergence, wrong target)?

Uses estimation-based (not analytical) methods.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from identification.config import ExperimentConfig, EstimationConfig
from identification.environment import SerializedContentEnvironment
from identification.estimators import (
    estimate_oracle,
    estimate_reduced_form_q,
    estimate_iq_learn,
    oracle_counterfactual_ccps,
)
from identification.counterfactuals import evaluate_type_ii, evaluate_type_ii_from_q
from identification.metrics import ccp_error


def main():
    cfg = ExperimentConfig()
    env = SerializedContentEnvironment(cfg.dgp)
    problem = env.problem_spec
    absorbing = cfg.dgp.absorbing_state
    skip_k = 3

    sample_sizes = [50, 100, 250, 500, 1000, 2000]

    new_trans = env.build_skip_transitions(skip_k)
    oracle_cf = oracle_counterfactual_ccps(env, new_transitions=new_trans)

    print(f"Sample size sweep: Type II CCP error (buy skips k={skip_k})")
    print(f"{'N':>6s}  {'RF-Q':>10s}  {'IQ-Learn':>10s}")

    results = []
    for N in sample_sizes:
        est_cfg = EstimationConfig(
            n_individuals=N,
            n_periods=80,
            seed=42,
            iq_max_iter=5000,
        )

        panel = env.generate_panel(
            n_individuals=N,
            n_periods=est_cfg.n_periods,
            seed=est_cfg.seed,
        )

        # Reduced-form Q
        t0 = time.time()
        rf = estimate_reduced_form_q(panel, env, est_cfg)
        rf_cf = evaluate_type_ii_from_q(rf["q_table"], problem.scale_parameter)
        rf_err = ccp_error(rf_cf, oracle_cf, absorbing)
        rf_time = time.time() - t0

        # IQ-Learn
        t0 = time.time()
        iq = estimate_iq_learn(panel, env, est_cfg)
        iq_cf = evaluate_type_ii(iq["reward_matrix"], new_trans, problem)
        iq_err = ccp_error(iq_cf, oracle_cf, absorbing)
        iq_time = time.time() - t0

        print(f"{N:6d}  {rf_err:10.6f}  {iq_err:10.6f}")
        results.append({
            "N": N,
            "rf_q_error": float(rf_err),
            "iq_learn_error": float(iq_err),
            "rf_q_time": rf_time,
            "iq_learn_time": iq_time,
        })

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "sample_size.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'sample_size.json'}")


if __name__ == "__main__":
    main()
