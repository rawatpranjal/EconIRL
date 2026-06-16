"""Forms loader: run an estimator roster on a Form and collect results.

``run_form`` is the load-bearing entry point.  Given a :class:`Form`, it:

1. Simulates a panel from the environment once.
2. Filters the estimator roster against the form's capability constraints.
3. Routes each surviving estimator through the correct call path:
   - ``path="estimate"``: low-level ``BaseEstimator.estimate(panel, spec, problem, T)``
   - ``path="fit_features"``: sklearn-style ``est.fit(panel, features=..., transitions=...)``
4. Wraps every run in try/except so one failure never crashes the whole run.
5. Returns a :class:`RunResult` with per-estimator results and a skip log.

Alias handling
--------------
GLADIUS and AIRL are aliases of NeuralGLADIUS and NeuralAIRL in the capability
registry (same underlying class, same path).  The loader skips the alias names
to avoid double-running: if NeuralGLADIUS is in the roster, GLADIUS is skipped
with reason ``"alias of NeuralGLADIUS (already run)"``, and likewise for AIRL.

Import hygiene
--------------
This module imports nothing from ``validation/`` or ``scripts/``.  All estimator
classes are imported inline (lazily) inside the private ``_build_estimator``
helper to avoid circular imports and keep the module load cheap.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from econirl.forms.base import Form
from econirl.forms.capabilities import CAPABILITIES, EstimatorCapability


# ---------------------------------------------------------------------------
# Public data carriers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Collected outputs from :func:`run_form`.

    Attributes:
        results: Mapping from estimator name to the result object returned by
            the estimator.  For ``path="estimate"`` estimators this is an
            :class:`~econirl.inference.results.EstimationSummary`; for
            ``path="fit_features"`` estimators this is the fitted estimator
            object (which exposes ``.policy_``, ``.value_``, ``.converged_``).
        skipped: List of ``{"name": str, "reason": str}`` records for every
            estimator that was not run (compatibility filter or alias skip or
            run-time error).
    """

    results: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: Aliases in the capability registry.  Skipped when their canonical
#: counterpart is present in the roster.
_ALIASES: dict[str, str] = {
    "AIRL": "NeuralAIRL",
    "GLADIUS": "NeuralGLADIUS",
}


def _spec_names(env) -> list[str]:
    """Parameter names for this environment.

    Uses the environment's own names when it has a finite-dimensional theta
    (linear reward).  Falls back to feature-count names when it does not
    (e.g. neural-reward Shapeshifter whose ``parameter_names`` is empty).
    Matches the convention in ``validation/benchmark/runner.py``.
    """
    names = list(env.parameter_names)
    if names:
        return names
    K = int(np.asarray(env.feature_matrix).shape[2])
    return [f"theta_{i}" for i in range(K)]


def _linear_utility(env):
    """Build a :class:`~econirl.preferences.linear.LinearUtility` from the env."""
    from econirl.preferences.linear import LinearUtility
    return LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=_spec_names(env),
    )


def _action_reward(env):
    """Build an :class:`~econirl.preferences.action_reward.ActionDependentReward`."""
    from econirl.preferences.action_reward import ActionDependentReward
    return ActionDependentReward(env.feature_matrix, _spec_names(env))


def _build_estimator(name: str, cap: EstimatorCapability, form: Form) -> Any:
    """Construct the estimator object for *name* using standard configs.

    Configs replicate ``scripts/quick_all_estimators.py`` so the loader
    behaves consistently with the benchmark harness.  This function imports
    estimator classes inline to keep the module load cheap.
    """
    num_states = form.spec.num_states

    if name == "NFXP":
        from econirl.estimation import NFXPEstimator
        return NFXPEstimator(
            inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=100_000,
            compute_hessian=True, verbose=False,
        )

    if name == "CCP":
        from econirl.estimation import CCPEstimator
        return CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=False)

    if name == "NNES":
        from econirl.estimation.nnes import NNESEstimator
        return NNESEstimator(
            hidden_dim=64, v_epochs=800, n_outer_iterations=5,
            compute_se=False, verbose=False,
        )

    if name == "SEES":
        from econirl.estimation.sees import SEESEstimator
        # The basis must span the value function (basis_dim >= num_states), so
        # the basis dimension tracks the state count rather than a fixed cap.
        basis_dim = num_states
        return SEESEstimator(
            basis_type="bspline", basis_dim=basis_dim,
            warm_start_value=True, penalty_weight=10.0,
            compute_se=False, verbose=False,
        )

    if name == "TDCCP":
        from econirl.estimation import TDCCPEstimator, TDCCPConfig
        return TDCCPEstimator(config=TDCCPConfig(
            hidden_dim=64, avi_iterations=15, epochs_per_avi=15,
            compute_se=False, verbose=False,
        ))

    if name == "UFXP":
        from econirl.estimation import UFXPEstimator
        return UFXPEstimator(weights="optimal", verbose=False)

    if name == "MCEIRL":
        from econirl.estimation import MCEIRLEstimator, MCEIRLConfig
        return MCEIRLEstimator(config=MCEIRLConfig(
            learning_rate=0.05, outer_max_iter=100,
            inner_max_iter=2000, compute_se=False, verbose=False,
        ))

    if name == "MaxEntIRL":
        from econirl.contrib.maxent_irl import MaxEntIRLEstimator
        return MaxEntIRLEstimator(
            inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
            learning_rate=0.05, compute_hessian=False, verbose=False,
        )

    if name == "MaxMarginIRL":
        from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator
        return MaxMarginIRLEstimator(
            max_iterations=50, margin_tol=1e-4,
            compute_hessian=False, verbose=False,
        )

    if name == "IQLearn":
        from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
        return IQLearnEstimator(config=IQLearnConfig(
            q_type="linear", divergence="chi2", alpha=3.0,
            max_iter=2000, verbose=False,
        ))

    if name == "MCEIRLNeural":
        from econirl.estimators import MCEIRLNeural
        ps = form.env.problem_spec
        return MCEIRLNeural(
            n_states=num_states,
            n_actions=form.spec.num_actions,
            discount=float(ps.discount_factor),
            max_epochs=200,
            verbose=False,
        )

    if name == "NeuralUFXP":
        from econirl.estimators import NeuralUFXP
        ps = form.env.problem_spec
        return NeuralUFXP(
            discount=float(ps.discount_factor),
            scale=float(getattr(ps, "scale_parameter", 1.0)),
            max_epochs=2000,
            verbose=False,
        )

    if name in ("NeuralAIRL", "AIRL"):
        from econirl.estimators import NeuralAIRL
        ps = form.env.problem_spec
        return NeuralAIRL(
            n_actions=form.spec.num_actions,
            discount=float(ps.discount_factor),
            max_epochs=300,
            verbose=False,
        )

    if name in ("NeuralGLADIUS", "GLADIUS"):
        from econirl.estimators import NeuralGLADIUS
        ps = form.env.problem_spec
        return NeuralGLADIUS(
            n_actions=form.spec.num_actions,
            discount=float(ps.discount_factor),
            max_epochs=300,
            verbose=False,
        )

    raise ValueError(f"run_form: no build recipe for estimator {name!r}")


def _run_estimate(
    estimator: Any,
    cap: EstimatorCapability,
    panel,
    form: Form,
) -> Any:
    """Run a ``path="estimate"`` estimator and return the EstimationSummary."""
    if cap.family == "structural":
        spec = _linear_utility(form.env)
    else:
        # Behavioral linear IRL: ActionDependentReward
        spec = _action_reward(form.env)

    return estimator.estimate(
        panel,
        spec,
        form.env.problem_spec,
        form.env.transition_matrices,
    )


def _run_fit_features(
    estimator: Any,
    panel,
    form: Form,
) -> Any:
    """Run a ``path="fit_features"`` estimator and return the fitted object.

    Model-free and model-based estimators take the same ``fit`` call; the
    model-free ones accept ``transitions`` and ignore it internally.
    """
    features = np.asarray(form.env.feature_matrix)
    transitions = np.asarray(form.env.transition_matrices)
    return estimator.fit(panel, features=features, transitions=transitions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_form(
    form: Form,
    estimators: list[str] | None = None,
    *,
    n_individuals: int = 300,
    n_periods: int = 40,
    seed: int = 0,
) -> RunResult:
    """Simulate a panel from *form* and run estimators against it.

    Args:
        form: A :class:`~econirl.forms.base.Form` wrapping a DDCEnvironment.
        estimators: Names from :data:`~econirl.forms.capabilities.CAPABILITIES`
            to run.  ``None`` runs all registered names.
        n_individuals: Panel individuals to simulate.
        n_periods: Periods per individual.
        seed: Simulation seed.

    Returns:
        A :class:`RunResult` with per-estimator results and a skip log.

    Raises:
        ValueError: If any name in *estimators* is not in the registry.
    """
    from econirl.simulation.synthetic import simulate_panel

    # ---- Validate estimator names ----
    if estimators is None:
        roster = list(CAPABILITIES.keys())
    else:
        unknown = [n for n in estimators if n not in CAPABILITIES]
        if unknown:
            raise ValueError(
                f"run_form: unknown estimator name(s): {unknown}.  "
                f"Valid names are: {sorted(CAPABILITIES.keys())}"
            )
        roster = list(estimators)

    # ---- Simulate once ----
    panel = simulate_panel(form.env, n_individuals, n_periods, seed=seed)

    result = RunResult()

    for name in roster:
        cap = CAPABILITIES[name]

        # --- Alias skip ---
        if name in _ALIASES:
            canonical = _ALIASES[name]
            if canonical in roster:
                result.skipped.append({
                    "name": name,
                    "reason": f"alias of {canonical} (already in roster; skipped to avoid double-run)",
                })
                continue

        # --- Compatibility checks ---
        if form.spec.reward_form not in cap.reward_forms:
            result.skipped.append({
                "name": name,
                "reason": (
                    f"form reward {form.spec.reward_form!r} not in supported "
                    f"{cap.reward_forms}"
                ),
            })
            continue

        if cap.needs_transitions and not form.spec.has_transitions:
            result.skipped.append({
                "name": name,
                "reason": (
                    f"estimator needs transition matrices but "
                    f"form.spec.has_transitions=False"
                ),
            })
            continue

        # --- Build and run ---
        try:
            est = _build_estimator(name, cap, form)
        except Exception as exc:
            result.skipped.append({
                "name": name,
                "reason": f"build error: {type(exc).__name__}: {exc}",
            })
            continue

        try:
            if cap.path == "estimate":
                res = _run_estimate(est, cap, panel, form)
            else:
                res = _run_fit_features(est, panel, form)
            result.results[name] = res
        except Exception as exc:  # noqa: BLE001 - failure is a data point
            result.skipped.append({
                "name": name,
                "reason": f"run error: {type(exc).__name__}: {exc}",
            })

    return result
