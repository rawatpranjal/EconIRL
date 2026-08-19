"""Estimator capability registry: which forms each public estimator can run.

The *support matrix* (``needs_transitions``, ``reward_forms``, ``model_free``,
``path``) is verified fact and drives how the loader routes each estimator. The
``group`` and ``shown_in_studies`` fields follow the curated headline roster used
by the simulation studies and are display curation, not routing.

Routing summary (verified):

* Structural (NFXP, CCP, NNES, SEES, TDCCP, UFXP) and linear IRL (MaxEntIRL,
  MaxMarginIRL, MCEIRL) and IQLearn: ``path="estimate"``, need transitions, linear.
* Neural model-based (MCEIRLNeural, NeuralUFXP): ``path="fit_features"``, need
  transitions, reward can be linear/nonlinear/neural.
* Tabular AIRL: ``path="fit_features"``, requires transitions and state-only
  linear rewards.
* NeuralAIRL: ``path="fit_features"``, requires transitions and accepts
  nonlinear neural state rewards.
* AIRL2: ``path="fit_features"``, requires transitions and linear
  segment-specific action-dependent rewards.
* Neural model-free (NeuralGLADIUS and the GLADIUS alias): ``path="fit_features"``,
  does not use transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

PATHS = ("estimate", "fit_features")
FAMILIES = ("structural", "behavioral")
GROUPS = ("classical", "modern", "causal-entropy", "model-free", "baseline", "other")
REWARD_FORMS = ("linear", "nonlinear", "neural")

_LINEAR = ("linear",)
_NEURAL = ("linear", "nonlinear", "neural")


@dataclass(frozen=True)
class EstimatorCapability:
    """What forms one public estimator supports, and how it is run and shown."""

    name: str
    family: str  # structural | behavioral (drives metric segmentation)
    needs_transitions: bool
    reward_forms: tuple[str, ...]
    model_free: bool
    generalizes_to_unvisited: bool
    path: str  # estimate | fit_features
    group: str  # display grouping in the studies
    shown_in_studies: bool

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: family must be in {FAMILIES}, got {self.family!r}")
        if self.path not in PATHS:
            raise ValueError(f"{self.name}: path must be in {PATHS}, got {self.path!r}")
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: group must be in {GROUPS}, got {self.group!r}")
        if not self.reward_forms or not set(self.reward_forms) <= set(REWARD_FORMS):
            raise ValueError(
                f"{self.name}: reward_forms must be a non-empty subset of {REWARD_FORMS}, "
                f"got {self.reward_forms!r}"
            )


def _structural(name: str, group: str, shown: bool) -> EstimatorCapability:
    return EstimatorCapability(
        name=name,
        family="structural",
        needs_transitions=True,
        reward_forms=_LINEAR,
        model_free=False,
        generalizes_to_unvisited=True,
        path="estimate",
        group=group,
        shown_in_studies=shown,
    )


def _linear_irl(
    name: str, group: str, shown: bool, generalizes: bool = True
) -> EstimatorCapability:
    return EstimatorCapability(
        name=name,
        family="behavioral",
        needs_transitions=True,
        reward_forms=_LINEAR,
        model_free=False,
        generalizes_to_unvisited=generalizes,
        path="estimate",
        group=group,
        shown_in_studies=shown,
    )


def _neural_model_based(name: str, group: str, shown: bool) -> EstimatorCapability:
    return EstimatorCapability(
        name=name,
        family="behavioral",
        needs_transitions=True,
        reward_forms=_NEURAL,
        model_free=False,
        generalizes_to_unvisited=True,
        path="fit_features",
        group=group,
        shown_in_studies=shown,
    )


def _linear_model_based_fit(name: str, group: str, shown: bool) -> EstimatorCapability:
    return EstimatorCapability(
        name=name,
        family="behavioral",
        needs_transitions=True,
        reward_forms=_LINEAR,
        model_free=False,
        generalizes_to_unvisited=True,
        path="fit_features",
        group=group,
        shown_in_studies=shown,
    )


def _neural_model_free(name: str, group: str, shown: bool) -> EstimatorCapability:
    return EstimatorCapability(
        name=name,
        family="behavioral",
        needs_transitions=False,
        reward_forms=_NEURAL,
        model_free=True,
        generalizes_to_unvisited=True,
        path="fit_features",
        group=group,
        shown_in_studies=shown,
    )


#: The verified capability matrix, keyed by public estimator class name.
#: Headline roster (shown_in_studies=True) is grouped by method family; everything
#: else is group "other" and not shown. Invariant: group=="other" iff not shown.
CAPABILITIES: dict[str, EstimatorCapability] = {
    c.name: c
    for c in (
        # Structural (finite linear theta; parameter recovery applies).
        _structural("NFXP", "classical", True),
        _structural("CCP", "classical", True),
        _structural("NNES", "modern", True),
        _structural("TDCCP", "modern", True),
        _structural("UFXP", "modern", True),
        _structural("SEES", "other", False),
        # Linear IRL (behavioral metrics; reward partially identified).
        _linear_irl("MCEIRL", "causal-entropy", True),
        # RHIP: horizon-parameterised MaxEnt-family IRL (Barnes et al. 2024).
        # H=inf reproduces MCE-IRL exactly; finite H interpolates to MMP (H=0).
        # Not in the curated headline roster (evidence lives in its own study).
        _linear_irl("RHIP", "other", False),
        _linear_irl("MaxEntIRL", "other", False),
        _linear_irl("MaxMarginIRL", "other", False),
        # Legacy / not identified: tabular Q, no generalization to unvisited states.
        _linear_irl("IQLearn", "other", False, generalizes=False),
        # Neural, model-based (fit(features=); transitions required).
        _neural_model_based("MCEIRLNeural", "causal-entropy", True),
        _neural_model_based("NeuralUFXP", "other", False),
        _linear_model_based_fit("AIRL", "causal-entropy", True),
        _neural_model_based("NeuralAIRL", "causal-entropy", True),
        # Anchored heterogeneous AIRL: linear segment rewards, transitions
        # required at fit, outside the headline studies roster.
        _linear_model_based_fit("AIRL2", "other", False),
        # Neural, model-free (fit(features=); transitions unused).
        _neural_model_free("NeuralGLADIUS", "model-free", True),
    )
}

# GLADIUS retains one compatibility name. Derive the entry from the canonical
# record so the two records cannot drift apart.
CAPABILITIES["GLADIUS"] = replace(CAPABILITIES["NeuralGLADIUS"], name="GLADIUS")
