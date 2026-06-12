"""Tests for estimator taxonomy and categorization."""

import pytest

from econirl.estimation.categories import (
    EstimatorCategory,
    ProblemCapabilities,
    ESTIMATOR_REGISTRY,
    CONTRIB_REGISTRY,
    get_estimators_by_category,
    get_estimators_with_capability,
    get_category,
    get_capabilities,
)


class TestEstimatorRegistry:
    """Tests for the estimator registry."""

    def test_production_estimators_registered(self):
        """14 production estimators (incl. UFXP, Neural MPEC, IQ-Learn)."""
        assert len(ESTIMATOR_REGISTRY) == 14

    def test_contrib_estimators_registered(self):
        """7 contrib estimators (IQ-Learn moved to production)."""
        assert len(CONTRIB_REGISTRY) == 7

    def test_known_estimators_present(self):
        """Key production estimators should be in the registry."""
        for name in ["NFXP", "CCP", "MPEC", "UFXP", "MCE IRL", "AIRL", "BC",
                      "GLADIUS", "IQ-Learn", "TD-CCP", "NNES", "SEES",
                      "Neural MPEC", "f-IRL"]:
            assert name in ESTIMATOR_REGISTRY, f"{name} not in registry"

    def test_contrib_estimators_present(self):
        """Moved estimators should be in contrib registry."""
        for name in ["MaxEnt IRL", "Deep MaxEnt", "GAIL",
                      "Max Margin", "GCL", "BIRL"]:
            assert name in CONTRIB_REGISTRY, f"{name} not in contrib registry"

    def test_registry_values_are_tuples(self):
        """Each registry entry should be (EstimatorCategory, ProblemCapabilities)."""
        for name, (cat, caps) in ESTIMATOR_REGISTRY.items():
            assert isinstance(cat, EstimatorCategory), f"{name} category is not EstimatorCategory"
            assert isinstance(caps, ProblemCapabilities), f"{name} capabilities is not ProblemCapabilities"


class TestEstimatorCategory:
    """Tests for category enumeration."""

    def test_structural_estimators(self):
        """NFXP, CCP, MPEC, UFXP should be structural."""
        structural = get_estimators_by_category(EstimatorCategory.STRUCTURAL)
        assert set(structural) == {"NFXP", "CCP", "MPEC", "UFXP"}

    def test_adversarial_estimators(self):
        """Only AIRL should be in production adversarial."""
        adversarial = get_estimators_by_category(EstimatorCategory.ADVERSARIAL_IRL)
        assert set(adversarial) == {"AIRL"}

    def test_q_learning_estimators(self):
        """GLADIUS and IQ-Learn should be in q_learning_irl."""
        q_learning = get_estimators_by_category(EstimatorCategory.Q_LEARNING_IRL)
        assert set(q_learning) == {"GLADIUS", "IQ-Learn"}

    def test_imitation_is_bc_only(self):
        """Only BC should be in the imitation category."""
        assert get_estimators_by_category(EstimatorCategory.IMITATION) == ["BC"]

    def test_structural_approx(self):
        """SEES, NNES, TD-CCP, Neural MPEC should be structural_approx."""
        approx = get_estimators_by_category(EstimatorCategory.STRUCTURAL_APPROX)
        assert set(approx) == {"SEES", "NNES", "TD-CCP", "Neural MPEC"}

    def test_entropy_irl(self):
        """MCE IRL should be entropy_irl."""
        entropy = get_estimators_by_category(EstimatorCategory.ENTROPY_IRL)
        assert set(entropy) == {"MCE IRL"}

    def test_distribution_irl(self):
        """f-IRL should be distribution_irl."""
        dist = get_estimators_by_category(EstimatorCategory.DISTRIBUTION_IRL)
        assert set(dist) == {"f-IRL"}


class TestProblemCapabilities:
    """Tests for capability-based filtering."""

    def test_bc_requires_no_transitions(self):
        """BC should be the only estimator not requiring transitions."""
        no_trans = get_estimators_with_capability(requires_transitions=False)
        assert no_trans == ["BC"]

    def test_structural_recover_params(self):
        """All structural estimators should recover structural params."""
        for name in ["NFXP", "CCP", "SEES", "NNES", "TD-CCP"]:
            caps = get_capabilities(name)
            assert caps.recovers_structural_params, f"{name} should recover params"

    def test_no_inner_solve_estimators(self):
        """CCP, SEES, NNES, TD-CCP, GLADIUS, BC should have no inner Bellman solve."""
        no_solve = get_estimators_with_capability(has_inner_bellman_solve=False)
        assert "CCP" in no_solve
        assert "GLADIUS" in no_solve
        assert "BC" in no_solve

    def test_scalable_estimators(self):
        """SEES, NNES, TD-CCP, GLADIUS, Neural MPEC support continuous states."""
        scalable = get_estimators_with_capability(supports_continuous_states=True)
        assert set(scalable) == {"SEES", "NNES", "TD-CCP", "GLADIUS", "Neural MPEC"}

    def test_finite_horizon_support(self):
        """NFXP, MCE IRL, AIRL should support finite horizon."""
        finite = get_estimators_with_capability(supports_finite_horizon=True)
        assert "NFXP" in finite
        assert "MCE IRL" in finite
        assert "AIRL" in finite

    def test_airl_uses_linear_reward(self):
        """AIRL should use linear reward type (benchmark config)."""
        caps = get_capabilities("AIRL")
        assert caps.reward_type == "linear"

    def test_multi_capability_filter(self):
        """Filtering by multiple capabilities should intersect."""
        result = get_estimators_with_capability(
            recovers_structural_params=True,
            supports_continuous_states=True,
        )
        assert set(result) == {"SEES", "NNES", "TD-CCP", "GLADIUS"}


class TestHelperFunctions:
    """Tests for get_category and get_capabilities."""

    def test_get_category_production(self):
        assert get_category("NFXP") == EstimatorCategory.STRUCTURAL

    def test_get_category_contrib(self):
        """Contrib estimators should also be accessible via get_category."""
        assert get_category("IQ-Learn") == EstimatorCategory.Q_LEARNING_IRL
        assert get_category("GAIL") == EstimatorCategory.ADVERSARIAL_IRL

    def test_get_capabilities(self):
        caps = get_capabilities("NFXP")
        assert caps.reward_type == "linear"
        assert caps.requires_transitions is True

    def test_unknown_estimator_raises(self):
        with pytest.raises(KeyError):
            get_category("NonExistent")
