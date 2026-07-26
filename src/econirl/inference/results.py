"""Estimation results with rich statistical inference.

This module provides the EstimationSummary class, which presents estimation
results in a StatsModels-style format with standard errors, confidence
intervals, and hypothesis tests.

The goal is to provide economists with familiar, publication-ready output
that matches the conventions of the structural estimation literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class IdentificationDiagnostics:
    """Diagnostics for parameter identification.

    Attributes:
        hessian_condition_number: Condition number of the Hessian matrix
        min_eigenvalue: Smallest eigenvalue of the Hessian
        max_eigenvalue: Largest eigenvalue of the Hessian
        rank: Numerical rank of the Hessian
        is_positive_definite: Whether Hessian is positive definite
        status: Human-readable identification status
    """

    hessian_condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    rank: int
    is_positive_definite: bool
    status: str


@dataclass
class GoodnessOfFit:
    """Goodness of fit measures.

    Attributes:
        log_likelihood: Maximized log-likelihood value
        num_parameters: Number of estimated parameters
        num_observations: Number of observations
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        pseudo_r_squared: McFadden's pseudo R-squared
        prediction_accuracy: Fraction of correctly predicted choices
    """

    log_likelihood: float
    num_parameters: int
    num_observations: int
    aic: float
    bic: float
    pseudo_r_squared: float | None = None
    prediction_accuracy: float | None = None


@dataclass
class DatasetInfo:
    """Dataset cardinality and panel-coverage statistics.

    ``obs_per_state`` holds the distribution of observation counts across the
    states that were actually visited, as a dict with keys
    ``max, p95, p50, p5, min``.
    """

    num_states: int
    num_actions: int
    num_observations: int
    num_individuals: int
    num_periods: int
    obs_per_state: dict[str, float]
    states_visited: int
    single_action_states: int


@dataclass
class PreEstimationChecks:
    """Identification-readiness checks.

    ``kind="feature"`` (structural, linear utility) carries the reward-design
    rank checks. ``kind="coverage"`` (IRL / neural, no linear feature matrix)
    carries panel-coverage statistics. Only the fields for the active ``kind``
    are populated.
    """

    kind: str
    # structural feature block
    num_features: int | None = None
    feature_rank: int | None = None
    condition_number: float | None = None
    contrast_rank: int | None = None
    contrast_condition_number: float | None = None
    verdict: str | None = None
    # IRL / neural coverage block (populated in a later pass)
    state_action_coverage: float | None = None
    state_coverage: float | None = None
    demo_policy_entropy: float | None = None
    effective_occupancy_support: float | None = None
    initial_states: int | None = None
    initial_state_entropy: float | None = None


@dataclass
class TransitionFirstStage:
    """First-stage state-transition estimate and its multinomial sampling error.

    The transition probabilities are estimated by empirical frequencies. Each
    estimated cell P(s'|s,a) carries the multinomial standard error
    ``sqrt(p_hat (1 - p_hat) / N_sa)`` where ``N_sa`` is the number of
    transitions observed out of ``(s, a)``. ``se_quantiles`` summarizes those
    standard errors across all estimated cells (keys ``max, p95, p50, p5, min``).
    ``probs`` / ``probs_se`` are populated only when the caller supplies a
    low-dimensional structured kernel worth listing.
    """

    method: str
    num_transitions: int
    num_free_parameters: int
    rows_with_support: int
    rows_total: int
    se_quantiles: dict[str, float]
    probs: list[float] | None = None
    probs_se: list[float] | None = None
    note: str = "Held fixed in stage two (block-diagonal information)."


def _quantiles(values: Any) -> dict[str, float]:
    """max, p95, p50, p5, min of a 1-D array; zeros for an empty input."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"max": 0.0, "p95": 0.0, "p50": 0.0, "p5": 0.0, "min": 0.0}
    return {
        "max": float(np.max(v)),
        "p95": float(np.percentile(v, 95)),
        "p50": float(np.percentile(v, 50)),
        "p5": float(np.percentile(v, 5)),
        "min": float(np.min(v)),
    }


def _coverage_checks(
    sa_counts: np.ndarray,
    panel: Any,
    num_states: int,
    num_actions: int,
) -> PreEstimationChecks:
    """Coverage-kind pre-estimation block for IRL / neural estimators.

    Reuses the ``(S, A)`` counts already accumulated in
    ``compute_fit_diagnostics`` -- no re-read of the panel's flat arrays.
    ``effective_occupancy_support`` needs transitions plus a policy and is left
    unset here.
    """
    obs_per_state = sa_counts.sum(axis=1)
    visited = obs_per_state > 0
    state_action_coverage = float((sa_counts > 0).sum()) / (num_states * num_actions)
    state_coverage = float(visited.sum()) / num_states

    total_obs = float(obs_per_state.sum())
    entropies: list[float] = []
    weights: list[float] = []
    for s in range(num_states):
        n_s = obs_per_state[s]
        if n_s == 0:
            continue
        p = sa_counts[s] / n_s
        p_nonzero = p[p > 0]
        entropies.append(float(-(p_nonzero * np.log(p_nonzero)).sum()))
        weights.append(float(n_s) / total_obs)
    demo_policy_entropy = float(np.average(entropies, weights=weights)) if entropies else 0.0

    first_states = np.array([int(traj.states[0]) for traj in panel.trajectories])
    _, init_counts = np.unique(first_states, return_counts=True)
    p_init = init_counts / init_counts.sum()
    initial_state_entropy = float(-(p_init * np.log(p_init)).sum())

    return PreEstimationChecks(
        kind="coverage",
        state_action_coverage=state_action_coverage,
        state_coverage=state_coverage,
        demo_policy_entropy=demo_policy_entropy,
        initial_states=int(init_counts.size),
        initial_state_entropy=initial_state_entropy,
    )


def compute_fit_diagnostics(
    panel: Any,
    num_states: int,
    num_actions: int,
    *,
    feature_matrix: Any | None = None,
) -> tuple[DatasetInfo, PreEstimationChecks | None, TransitionFirstStage | None]:
    """Compute the DATA, PRE-ESTIMATION, and FIRST-STAGE TRANSITION blocks.

    Reads the panel once. ``feature_matrix`` of shape ``(S, A, K)`` selects the
    structural feature-rank pre-estimation block; when ``None`` a coverage-kind
    block is returned instead (panel-coverage statistics, no linear reward
    design to check).

    Counts are accumulated sparsely (via ``np.unique`` over observed tuples), so
    this scales to large state spaces without materializing an ``(A, S, S)``
    tensor.
    """
    states = np.asarray(panel.get_all_states()).astype(np.int64)
    actions = np.asarray(panel.get_all_actions()).astype(np.int64)
    next_states = np.asarray(panel.get_all_next_states()).astype(np.int64)

    # --- (S, A) counts -> dataset / coverage stats ---
    sa_counts = np.zeros((num_states, num_actions), dtype=np.int64)
    np.add.at(sa_counts, (states, actions), 1)
    obs_per_state = sa_counts.sum(axis=1)
    visited = obs_per_state > 0
    actions_per_state = (sa_counts > 0).sum(axis=1)
    single_action_states = int((actions_per_state == 1).sum())

    dataset = DatasetInfo(
        num_states=num_states,
        num_actions=num_actions,
        num_observations=int(panel.num_observations),
        num_individuals=int(panel.num_individuals),
        num_periods=int(max(panel.num_periods_per_individual)),
        obs_per_state=_quantiles(obs_per_state[visited]),
        states_visited=int(visited.sum()),
        single_action_states=single_action_states,
    )

    # --- structural feature-rank block ---
    pre_estimation: PreEstimationChecks | None = None
    if feature_matrix is not None:
        from econirl.preprocessing.diagnostics import feature_diagnostics

        fd = feature_diagnostics(np.asarray(feature_matrix))
        k = int(fd["num_features"])
        contrast_rank = int(fd["contrast_rank"])
        identified = contrast_rank >= k
        verdict = (
            "identified -- every reward parameter varies across actions"
            if identified
            else f"under-identified -- contrast rank {contrast_rank} < {k} features"
        )
        pre_estimation = PreEstimationChecks(
            kind="feature",
            num_features=k,
            feature_rank=int(fd["feature_rank"]),
            condition_number=float(fd["condition_number"]),
            contrast_rank=contrast_rank,
            contrast_condition_number=float(fd["contrast_condition_number"]),
            verdict=verdict,
        )
    else:
        pre_estimation = _coverage_checks(sa_counts, panel, num_states, num_actions)

    # --- first-stage transition estimate + multinomial SE (sparse) ---
    transition_first_stage: TransitionFirstStage | None = None
    n_trans = int(states.shape[0])
    if n_trans > 0:
        triples = np.stack([actions, states, next_states], axis=1)
        uniq_triples, tri_counts = np.unique(triples, axis=0, return_counts=True)
        pairs = np.stack([actions, states], axis=1)
        uniq_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        pair_to_n = {(int(a), int(s)): int(n) for (a, s), n in zip(uniq_pairs, pair_counts)}
        ses: list[float] = []
        support_per_pair: dict[tuple[int, int], int] = {}
        for (a, s, _sp), c in zip(uniq_triples, tri_counts):
            n_sa = pair_to_n[(int(a), int(s))]
            p = float(c) / n_sa
            ses.append(float(np.sqrt(p * (1.0 - p) / n_sa)))
            key = (int(a), int(s))
            support_per_pair[key] = support_per_pair.get(key, 0) + 1
        free_params = sum(max(k - 1, 0) for k in support_per_pair.values())
        transition_first_stage = TransitionFirstStage(
            method="empirical frequencies (multinomial MLE)",
            num_transitions=n_trans,
            num_free_parameters=int(free_params),
            rows_with_support=int(len(uniq_pairs)),
            rows_total=int(num_states * num_actions),
            se_quantiles=_quantiles(ses),
        )

    return dataset, pre_estimation, transition_first_stage


@dataclass
class EstimationSummary:
    """Rich estimation results with statistical inference.

    This class provides a StatsModels-style interface for presenting
    estimation results, including:
    - Point estimates with standard errors
    - Confidence intervals
    - Hypothesis tests (t-tests, Wald tests)
    - Identification diagnostics
    - Goodness of fit measures
    - Publication-ready output (summary tables, LaTeX)

    Attributes:
        parameters: Estimated parameter values
        parameter_names: Names of parameters
        standard_errors: Standard errors of estimates
        hessian: Hessian matrix at optimum (for inference)
        method: Estimation method used
        convergence_info: Details about optimization convergence

    Example:
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.summary())
        >>> result.to_latex("table.tex")
    """

    # Core estimates
    parameters: jnp.ndarray
    parameter_names: list[str]
    standard_errors: jnp.ndarray

    # Inference components
    hessian: jnp.ndarray | None = None
    variance_covariance: jnp.ndarray | None = None

    # Model info
    method: str = "Unknown"
    num_observations: int = 0
    num_individuals: int = 0
    num_periods: int = 0

    # Structural parameters
    discount_factor: float = 0.9999
    scale_parameter: float = 1.0

    # Fit and diagnostics
    log_likelihood: float | None = None
    goodness_of_fit: GoodnessOfFit | None = None
    identification: IdentificationDiagnostics | None = None

    # Convergence
    converged: bool = True
    num_iterations: int = 0
    convergence_message: str = ""

    # Solution
    value_function: jnp.ndarray | None = None
    policy: jnp.ndarray | None = None

    # Metadata
    estimation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    # Expanded diagnostics (data / pre-estimation / first-stage transition)
    num_states: int | None = None
    num_actions: int | None = None
    optimizer: str | None = None
    transition_source: str | None = None
    dataset: DatasetInfo | None = None
    pre_estimation: PreEstimationChecks | None = None
    transition_first_stage: TransitionFirstStage | None = None

    # Oracle (known-truth; sim studies). When set, the parameter table gains
    # true / bias / |err| columns. Monte-Carlo oracle uses MonteCarloResult.
    true_parameters: dict[str, float] | None = None
    oracle_policy: jnp.ndarray | None = None
    oracle_value: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate and compute derived quantities."""
        if len(self.parameters) != len(self.parameter_names):
            raise ValueError(
                f"parameters ({len(self.parameters)}) and parameter_names "
                f"({len(self.parameter_names)}) must have same length"
            )

        # Compute variance-covariance from Hessian if not provided
        if self.variance_covariance is None and self.hessian is not None:
            try:
                self.variance_covariance = jnp.linalg.inv(-self.hessian)
            except Exception:
                pass  # Hessian not invertible

    @property
    def num_parameters(self) -> int:
        """Number of estimated parameters."""
        return len(self.parameters)

    @property
    def t_statistics(self) -> jnp.ndarray:
        """T-statistics for each parameter (H0: θ = 0)."""
        return self.parameters / self.standard_errors

    @property
    def p_values(self) -> jnp.ndarray:
        """Two-sided p-values for t-tests (H0: θ = 0)."""
        t_stats = np.asarray(self.t_statistics)
        # Large sample: use normal approximation
        p_vals = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        return jnp.array(p_vals, dtype=jnp.float32)

    def confidence_interval(self, alpha: float = 0.05) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute confidence intervals for parameters.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * self.standard_errors
        lower = self.parameters - margin
        upper = self.parameters + margin
        return lower, upper

    def get_parameter(self, name: str) -> dict[str, float]:
        """Get detailed results for a single parameter.

        Args:
            name: Parameter name

        Returns:
            Dictionary with estimate, se, t-stat, p-value, CI
        """
        idx = self.parameter_names.index(name)
        lower, upper = self.confidence_interval()

        return {
            "estimate": float(self.parameters[idx]),
            "std_error": float(self.standard_errors[idx]),
            "t_statistic": float(self.t_statistics[idx]),
            "p_value": float(self.p_values[idx]),
            "ci_lower": float(lower[idx]),
            "ci_upper": float(upper[idx]),
        }

    def wald_test(
        self,
        R: jnp.ndarray,
        r: jnp.ndarray | None = None,
    ) -> dict[str, float]:
        """Perform a Wald test for linear restrictions.

        Tests H0: R @ θ = r against H1: R @ θ ≠ r

        Args:
            R: Restriction matrix of shape (num_restrictions, num_parameters)
            r: Restriction values of shape (num_restrictions,). Default is zeros.

        Returns:
            Dictionary with test statistic, degrees of freedom, and p-value
        """
        if r is None:
            r = jnp.zeros(R.shape[0])

        if self.variance_covariance is None:
            raise ValueError("Variance-covariance matrix required for Wald test")

        # Wald statistic: (Rθ - r)' [R V R']^{-1} (Rθ - r)
        diff = R @ self.parameters - r
        middle = R @ self.variance_covariance @ R.T
        wald_stat = float(diff @ jnp.linalg.inv(middle) @ diff)

        df = R.shape[0]
        p_value = 1 - stats.chi2.cdf(wald_stat, df)

        return {
            "statistic": wald_stat,
            "df": df,
            "p_value": p_value,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns: estimate, std_error, t_stat, p_value, ci_lower, ci_upper
        """
        lower, upper = self.confidence_interval()

        return pd.DataFrame(
            {
                "estimate": np.asarray(self.parameters),
                "std_error": np.asarray(self.standard_errors),
                "t_statistic": np.asarray(self.t_statistics),
                "p_value": np.asarray(self.p_values),
                "ci_lower": np.asarray(lower),
                "ci_upper": np.asarray(upper),
            },
            index=self.parameter_names,
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a StatsModels-style summary table.

        Args:
            alpha: Significance level for confidence intervals

        Returns:
            Formatted string with estimation results
        """
        width = 80
        sep = "=" * width
        dash = "-" * width

        lines = [
            sep,
            "Dynamic Discrete Choice Estimation Results".center(width),
            sep,
        ]

        # --- header block ---
        info_left = [f"Method:      {self.method}"]
        if self.optimizer:
            info_left.append(f"Optimizer:   {self.optimizer}")
        family = self._family_label()
        if family:
            info_left.append(f"Family:      {family}")
        info_right = [
            f"Observations:  {self.num_observations:,}",
            f"Individuals:   {self.num_individuals:,}",
            f"Discount (β):  {self.discount_factor}",
            f"Scale (σ):     {self.scale_parameter}",
            f"Date:          {self.timestamp[:10]}",
        ]
        label_width = max(len(left) for left in info_left) + 2
        for i in range(max(len(info_left), len(info_right))):
            left = info_left[i] if i < len(info_left) else ""
            right = info_right[i] if i < len(info_right) else ""
            lines.append(f"{left:<{label_width}}{right}".rstrip())

        # --- [1] DATA ---
        if self.dataset is not None:
            d = self.dataset
            q = d.obs_per_state
            lines.append("")
            lines.append("[1] DATA")
            lines.append(f"  State space:          {d.num_states} states x {d.num_actions} actions")
            lines.append(f"  Periods per individual: ~{d.num_periods}")
            lines.append(
                "  Obs per state:        "
                f"max {q['max']:,.0f} . p95 {q['p95']:,.0f} . p50 {q['p50']:,.0f}"
                f" . p5 {q['p5']:,.0f} . min {q['min']:,.0f}"
            )
            pct = 100.0 * d.states_visited / d.num_states if d.num_states else 0.0
            lines.append(
                f"  State coverage:       {d.states_visited}/{d.num_states} visited ({pct:.0f}%)"
            )
            lines.append(f"  Single-action states: {d.single_action_states}")

        # --- [2] PRE-ESTIMATION CHECKS ---
        if self.pre_estimation is not None:
            lines.append("")
            lines.append("[2] PRE-ESTIMATION CHECKS")
            lines.extend(self._render_pre_estimation(self.pre_estimation))

        # --- [3] TRANSITION MODEL ---
        if self.transition_source is not None or self.transition_first_stage is not None:
            lines.append("")
            if self.transition_first_stage is None:
                lines.append("[3] TRANSITION MODEL")
            else:
                lines.append("[3] FIRST-STAGE TRANSITION ESTIMATION")
            if self.transition_source is not None:
                lines.append(f"  Transition source: {self.transition_source}")

        if self.transition_first_stage is not None:
            t = self.transition_first_stage
            q = t.se_quantiles
            lines.append(f"  Method:               {t.method}")
            lines.append(
                f"  Transitions used:     N = {t.num_transitions:,}"
                f"        Free parameters: {t.num_free_parameters:,}"
            )
            lines.append(f"  Rows with full support: {t.rows_with_support}/{t.rows_total}")
            lines.append(
                "  Std err across cells: "
                f"max {q['max']:.4f} . p50 {q['p50']:.4f} . min {q['min']:.4f}"
            )
            if t.probs is not None and t.probs_se is not None:
                cells = "   ".join(
                    f"p[{k}] {p:.4f} (se {s:.4f})"
                    for k, (p, s) in enumerate(zip(t.probs, t.probs_se))
                )
                lines.append(f"    {cells}")
            lines.append(f"  {t.note}")

        lines.append(dash)

        # --- [4] RESULTS ---
        lines.append("[4] RESULTS")
        lines.append("  4a. Estimation")
        lines.extend(self._render_parameter_table(alpha))

        if self.identification is not None:
            lines.append("  4b. Identification")
            lines.append(
                "    Hessian condition:  "
                f"{self.identification.hessian_condition_number:,.1f}"
                f"     Min eigenvalue: {self.identification.min_eigenvalue:.2f}"
            )
            lines.append(f"    Status:             {self.identification.status}")

        lines.append("  4c. Inference & fit")
        lines.append(f"    Converged:   {'yes' if self.converged else 'no'}")
        lines.append(f"    Iterations:  {self.num_iterations}")
        lines.append(f"    Estimation time: {self.estimation_time:.2f} seconds")
        if self.convergence_message:
            lines.append(f"    Message:     {self.convergence_message}")
        parameter_residual = self.metadata.get("npl_parameter_residual")
        policy_residual = self.metadata.get("npl_policy_residual")
        tolerance = self.metadata.get("npl_convergence_tolerance")
        if parameter_residual is not None and policy_residual is not None:
            lines.append(
                "    NPL residuals: "
                f"parameter {float(parameter_residual):.3e}, "
                f"policy {float(policy_residual):.3e}"
            )
            if tolerance is not None:
                lines.append(f"    NPL tolerance: {float(tolerance):.3e}")
        se_label = self._se_method_label()
        if se_label:
            lines.append(f"    SE method:  {se_label}")
        if self.log_likelihood is not None:
            lines.append(f"    Log-lik:    {self.log_likelihood:,.2f}")
        if self.goodness_of_fit is not None:
            gof = self.goodness_of_fit
            lines.append(f"    AIC/BIC:    {gof.aic:,.1f} / {gof.bic:,.1f}")
            extras = []
            if gof.pseudo_r_squared is not None:
                extras.append(f"pseudo R2 {gof.pseudo_r_squared:.3f}")
            if gof.prediction_accuracy is not None:
                extras.append(f"accuracy {gof.prediction_accuracy:.1%}")
            if extras:
                lines.append("    " + " / ".join(extras))

        lines.append(sep)

        return "\n".join(lines)

    def _family_label(self) -> str | None:
        """Estimator family, inferred from the pre-estimation block kind."""
        if self.pre_estimation is None:
            return None
        if self.pre_estimation.kind == "feature":
            return "structural (linear utility)"
        if self.pre_estimation.kind == "coverage":
            return "IRL / neural"
        return None

    def _se_method_label(self) -> str | None:
        """Human-readable standard-error method from metadata."""
        method = self.metadata.get("se_method")
        if not isinstance(method, str):
            return None
        labels = {
            "asymptotic": "asymptotic (observed information)",
            "robust": "robust (sandwich)",
            "bootstrap": "bootstrap (resampled individuals)",
            "clustered": "clustered (by individual)",
            "full_likelihood_bhhh": "BHHH (outer-product of gradients)",
        }
        return labels.get(method, method)

    def _render_pre_estimation(self, pre: PreEstimationChecks) -> list[str]:
        """Render the structural feature block or the coverage block."""
        out: list[str] = []
        if pre.kind == "feature":
            out.append(f"  Reward features (K):  {pre.num_features}")
            out.append(
                f"  Design rank:          {pre.feature_rank}/{pre.num_features}"
                f"         Condition:          {pre.condition_number:.1e}"
            )
            out.append(
                f"  Contrast rank:        {pre.contrast_rank}/{pre.num_features}"
                f"       Contrast condition: {pre.contrast_condition_number:.1e}"
            )
            out.append(f"  Verdict:              {pre.verdict}")
        elif pre.kind == "coverage":
            if pre.state_action_coverage is not None:
                out.append(f"  State-action coverage: {pre.state_action_coverage:.1%}")
            if pre.state_coverage is not None:
                out.append(f"  State coverage:        {pre.state_coverage:.1%}")
            if pre.demo_policy_entropy is not None:
                out.append(f"  Demo policy entropy:   {pre.demo_policy_entropy:.2f} nats")
            if pre.effective_occupancy_support is not None:
                out.append(f"  Occupancy support:     eff. {pre.effective_occupancy_support:.1f}")
            if pre.initial_states is not None:
                ent = (
                    f" (entropy {pre.initial_state_entropy:.2f} nats)"
                    if pre.initial_state_entropy is not None
                    else ""
                )
                out.append(f"  Initial-state coverage: {pre.initial_states} states{ent}")
        return out

    def _render_parameter_table(self, alpha: float) -> list[str]:
        """Parameter rows, with true / bias / |err| columns when an oracle is set."""
        out: list[str] = []
        lower, upper = self.confidence_interval(alpha)
        true_params = self.true_parameters
        if true_params is not None:
            header = f"    {'':16}{'coef':>10}{'std err':>10}{'true':>10}{'bias':>10}{'|err|':>9}"
        else:
            header = (
                f"    {'':16}{'coef':>10}{'std err':>10}{'t':>8}"
                f"{'P>|t|':>8}{'[0.025':>9}{'0.975]':>9}"
            )
        out.append(header)
        for i, name in enumerate(self.parameter_names):
            coef = float(self.parameters[i])
            se = float(self.standard_errors[i])
            if true_params is not None:
                true_val = true_params.get(name)
                if true_val is None:
                    row = f"    {name:16}{coef:>10.4f}{se:>10.4f}{'--':>10}{'--':>10}{'--':>9}"
                else:
                    bias = coef - true_val
                    row = (
                        f"    {name:16}{coef:>10.4f}{se:>10.4f}{true_val:>10.4f}"
                        f"{bias:>+10.4f}{abs(bias):>9.4f}"
                    )
            else:
                t = float(self.t_statistics[i])
                p = float(self.p_values[i])
                p_str = "0.000" if p < 0.001 else f"{p:.3f}"
                row = (
                    f"    {name:16}{coef:>10.4f}{se:>10.4f}{t:>8.2f}"
                    f"{p_str:>8}{float(lower[i]):>9.4f}{float(upper[i]):>9.4f}"
                )
            out.append(row)
        return out

    def to_latex(
        self,
        filename: str | None = None,
        caption: str = "Estimation Results",
        label: str = "tab:estimation",
    ) -> str:
        """Generate a LaTeX table of results.

        Args:
            filename: If provided, write to this file
            caption: Table caption
            label: Table label for referencing

        Returns:
            LaTeX table as string
        """
        df = self.to_dataframe()

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lcccccc}",
            r"\hline\hline",
            (
                r"Parameter & Estimate & Std. Error & $t$-stat & $p$-value & "
                r"\multicolumn{2}{c}{95\% CI} \\"
            ),
            r"\hline",
        ]

        for name in self.parameter_names:
            row = df.loc[name]
            lines.append(
                f"{name} & {row['estimate']:.4f} & {row['std_error']:.4f} & "
                f"{row['t_statistic']:.2f} & {row['p_value']:.3f} & "
                f"[{row['ci_lower']:.4f}, & {row['ci_upper']:.4f}] \\\\"
            )

        lines.extend(
            [
                r"\hline\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        latex = "\n".join(lines)

        if filename is not None:
            with open(filename, "w") as f:
                f.write(latex)

        return latex

    def diagnostics(self) -> dict[str, Any]:
        """Bundle all available diagnostics into a single dict.

        Returns a dict with sections for goodness of fit, identification,
        numerical quality, and convergence. Each section contains the
        diagnostics that are available for this estimation result.
        Missing diagnostics are omitted rather than set to None.

        Returns:
            Dict with section keys: goodness_of_fit, identification,
            numerical_quality, convergence.
        """
        result: dict[str, Any] = {}

        # Goodness of fit
        if self.goodness_of_fit is not None:
            gof = self.goodness_of_fit
            result["goodness_of_fit"] = {
                "log_likelihood": gof.log_likelihood,
                "aic": gof.aic,
                "bic": gof.bic,
                "num_parameters": gof.num_parameters,
                "num_observations": gof.num_observations,
            }
            if gof.pseudo_r_squared is not None:
                result["goodness_of_fit"]["pseudo_r_squared"] = gof.pseudo_r_squared
            if gof.prediction_accuracy is not None:
                result["goodness_of_fit"]["prediction_accuracy"] = gof.prediction_accuracy

        # Identification
        if self.identification is not None:
            ident = self.identification
            result["identification"] = {
                "condition_number": ident.hessian_condition_number,
                "min_eigenvalue": ident.min_eigenvalue,
                "max_eigenvalue": ident.max_eigenvalue,
                "rank": ident.rank,
                "is_positive_definite": ident.is_positive_definite,
                "status": ident.status,
            }

        # Numerical quality
        num_quality: dict[str, Any] = {}
        if self.hessian is not None:
            eigenvalues = np.sort(np.real(np.linalg.eigvals(np.asarray(-self.hessian))))
            num_quality["hessian_eigenvalues"] = eigenvalues.tolist()
            num_quality["hessian_condition_number"] = (
                float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf")
            )
        num_quality["converged"] = self.converged
        num_quality["num_iterations"] = self.num_iterations
        num_quality["convergence_message"] = self.convergence_message
        num_quality["estimation_time"] = self.estimation_time
        if num_quality:
            result["numerical_quality"] = num_quality

        # Convergence
        result["convergence"] = {
            "converged": self.converged,
            "num_iterations": self.num_iterations,
            "message": self.convergence_message,
        }

        return result

    def __repr__(self) -> str:
        return (
            f"EstimationSummary(method='{self.method}', "
            f"n_params={self.num_parameters}, "
            f"converged={self.converged})"
        )

    def __str__(self) -> str:
        return self.summary()
