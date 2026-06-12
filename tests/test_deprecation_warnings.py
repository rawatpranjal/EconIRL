"""Test that legacy API imports emit DeprecationWarning."""
import pytest


def test_nfxp_estimator_deprecation_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from econirl import NFXPEstimator
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "NFXPEstimator import should emit DeprecationWarning"
        assert "NFXP" in str(dep_warnings[0].message)


def test_ccp_estimator_deprecation_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from econirl import CCPEstimator
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "CCPEstimator import should emit DeprecationWarning"
        assert "CCP" in str(dep_warnings[0].message)


def test_mpec_slsqp_alias_deprecation_warning():
    """MPECConfig(solver='slsqp') warns and points at solver='sqp'."""
    import warnings
    from econirl.estimation.mpec import MPECConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MPECConfig(solver="slsqp")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "solver='slsqp' should emit DeprecationWarning"
        assert "sqp" in str(dep_warnings[0].message)


def test_mpec_sqp_no_deprecation_warning():
    """The recommended solver='sqp' path emits no deprecation warning."""
    import warnings
    from econirl.estimation.mpec import MPECConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MPECConfig(solver="sqp")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0, "solver='sqp' should not warn"
