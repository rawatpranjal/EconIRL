"""Statistical inference for estimation results."""

from econirl.inference.bootstrap import warm_start_bootstrap
from econirl.inference.etable import etable
from econirl.inference.fit_metrics import (
    brier_score,
    ccp_consistency_test,
    efron_pseudo_r_squared,
    kl_divergence,
)
from econirl.inference.hypothesis_tests import (
    likelihood_ratio_test,
    score_test,
    vuong_test,
)
from econirl.inference.identifiability import check_reward_identifiability
from econirl.inference.identification import check_identification
from econirl.inference.profile_likelihood import profile_likelihood
from econirl.inference.results import BootstrapResult, EstimationSummary
from econirl.inference.reward_comparison import (
    detect_reward_shaping,
    epic_distance,
)
from econirl.inference.sensitivity import discount_factor_sensitivity
from econirl.inference.standard_errors import compute_standard_errors

__all__ = [
    "EstimationSummary",
    "BootstrapResult",
    "compute_standard_errors",
    "check_identification",
    # Hypothesis tests
    "likelihood_ratio_test",
    "score_test",
    "vuong_test",
    # Fit metrics
    "brier_score",
    "kl_divergence",
    "efron_pseudo_r_squared",
    "ccp_consistency_test",
    # Reward comparison
    "epic_distance",
    "detect_reward_shaping",
    # Tables
    "etable",
    # Tier 3: advanced diagnostics
    "check_reward_identifiability",
    "discount_factor_sensitivity",
    "warm_start_bootstrap",
    "profile_likelihood",
]
