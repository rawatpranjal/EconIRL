"""Deprecated AIRL-Het compatibility module.

Use :mod:`econirl.estimation.adversarial.airl2` for new code. The legacy names
remain readable through the 0.1.x series so existing imports and pickles keep
working, but each lookup emits a deprecation warning.
"""

from __future__ import annotations

import warnings

from econirl.estimation.adversarial.airl2 import AIRL2Config, AIRL2Estimator

__all__ = ["AIRLHetConfig", "AIRLHetEstimator"]  # noqa: F822

_ALIASES = {
    "AIRLHetConfig": AIRL2Config,
    "AIRLHetEstimator": AIRL2Estimator,
}


def __getattr__(name: str):
    replacement = _ALIASES.get(name)
    if replacement is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{name} is deprecated; use {replacement.__name__} from "
        "econirl.estimation.adversarial.airl2. The AIRLHet alias will be "
        "removed after the 0.1.x series.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement
