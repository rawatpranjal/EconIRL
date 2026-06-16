"""Synthetic-DGP form specifications.

A *form* is a known-truth dynamic-choice problem in a standard state topology
(tabular, grid, graph, factored) crossed with a reward form (linear, nonlinear,
neural). ``FormSpec`` is the metadata; ``Form`` wraps an existing
``DDCEnvironment`` and never rewrites it. The factories in ``tabular.py`` /
``grid.py`` / ``graph.py`` / ``factored.py`` (added later) produce ``Form``
objects; this module only defines the data carriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the env layer (and JAX) at module load
    from econirl.environments.base import DDCEnvironment

#: Supported state topologies.
TOPOLOGIES = ("tabular", "grid", "graph", "factored")

#: Supported reward forms, orthogonal to topology.
REWARD_FORMS = ("linear", "nonlinear", "neural")


@dataclass(frozen=True)
class FormSpec:
    """Metadata describing a synthetic form: a topology crossed with a reward form."""

    topology: str
    reward_form: str
    num_states: int
    num_actions: int
    has_transitions: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {TOPOLOGIES}, got {self.topology!r}"
            )
        if self.reward_form not in REWARD_FORMS:
            raise ValueError(
                f"reward_form must be one of {REWARD_FORMS}, got {self.reward_form!r}"
            )

    @property
    def has_finite_theta(self) -> bool:
        """True only for a linear reward.

        A linear reward has a finite parameter vector, so parameter bias and
        coverage metrics apply to it. Nonlinear and neural rewards do not, which
        is what the metric segmentation keys on.
        """
        return self.reward_form == "linear"


@dataclass(frozen=True)
class Form:
    """A known-truth form: a :class:`FormSpec` plus the env that realizes it.

    The env is wrapped, not rewritten. Common attributes are delegated so a
    ``Form`` can stand in where an environment's arrays are needed.
    """

    spec: FormSpec
    env: "DDCEnvironment"

    @property
    def transition_matrices(self):  # noqa: ANN201 - delegated env attr
        return self.env.transition_matrices

    @property
    def feature_matrix(self):  # noqa: ANN201
        return self.env.feature_matrix

    @property
    def true_parameters(self):  # noqa: ANN201
        return self.env.true_parameters

    @property
    def parameter_names(self):  # noqa: ANN201
        return self.env.parameter_names

    @property
    def problem_spec(self):  # noqa: ANN201
        return self.env.problem_spec
