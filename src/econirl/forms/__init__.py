"""Synthetic-DGP forms framework (internal).

Known-truth dynamic-choice problems in standard topologies, plus the estimator
capability registry that says which forms each estimator can run, the topology
factories that build Form objects, and the run_form loader that executes an
estimator roster against a form in a single call.

This subpackage is not yet part of the public ``econirl`` API surface;
the public exports are added in a later chunk.
"""

from econirl.forms.base import REWARD_FORMS, TOPOLOGIES, Form, FormSpec
from econirl.forms.capabilities import CAPABILITIES, EstimatorCapability
from econirl.forms.factored import factored
from econirl.forms.grid import grid
from econirl.forms.loader import RunResult, run_form
from econirl.forms.tabular import tabular


def make_form(topology: str, reward_form: str = "linear", **kw) -> Form:
    """Route a topology + reward_form pair to the matching factory.

    Args:
        topology: One of ``"tabular"``, ``"grid"``, ``"factored"``.
            ``"graph"`` is reserved for a later chunk and raises
            :exc:`ValueError` now.
        reward_form: ``"linear"``, ``"nonlinear"``, or ``"neural"``.
        **kw: Forwarded to the topology factory unchanged.

    Returns:
        A :class:`Form` wrapping the appropriate DDCEnvironment.

    Raises:
        ValueError: For ``topology="graph"`` (not yet implemented),
            an unknown topology, or an unsupported (topology, reward_form)
            combo.
    """
    if topology == "tabular":
        return tabular(reward_form=reward_form, **kw)
    if topology == "grid":
        return grid(reward_form=reward_form, **kw)
    if topology == "factored":
        return factored(reward_form=reward_form, **kw)
    if topology == "graph":
        raise ValueError(
            "topology='graph' is reserved for a future chunk (F2: road_network "
            "graph generator) and is not yet implemented."
        )
    raise ValueError(
        f"make_form: unknown topology {topology!r}.  "
        f"Supported: {list(TOPOLOGIES)!r}"
    )


__all__ = [
    # Base data carriers
    "Form",
    "FormSpec",
    "TOPOLOGIES",
    "REWARD_FORMS",
    # Capability registry
    "EstimatorCapability",
    "CAPABILITIES",
    # Topology factories
    "tabular",
    "grid",
    "factored",
    # Dispatcher
    "make_form",
    # Loader
    "RunResult",
    "run_form",
]
