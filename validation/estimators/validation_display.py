"""Human-facing labels and context for known-truth validation cells."""

from __future__ import annotations

VALIDATION_DISPLAY_NAMES: dict[str, str] = {
    "canonical_low_action": "low-dimensional action-dependent DGP",
    "canonical_high_action": "high-dimensional action-dependent DGP",
    "canonical_low_state_only": "state-only diagnostic DGP",
    "canonical_latent_segments": "large latent-segment diagnostic DGP",
    "mce_low_high_reward": "low-state high-reward-feature DGP",
    "airl_paper_identification": "original AIRL identification DGP",
    "airl_anchor_action_dependent": "anchored action-dependent AIRL diagnostic DGP",
    "airl_het_paper_identification": "latent-segment AIRL-Het DGP",
    "f_irl_paper_state_marginal": "state-marginal f-IRL DGP",
    "gladius_paper_high_state": "high-dimensional state GLADIUS DGP",
    "gladius_paper_high_state_scaled": "scaled high-dimensional state GLADIUS DGP",
    "deep_mce_neural_reward": "anchored nonlinear reward-map DGP",
    "deep_mce_neural_features": "neural-feature projection diagnostic DGP",
    "deep_mce_neural_reward_features": "anchored nonlinear reward-and-feature DGP",
    "shapeshifter_linear_reward_neural_features": "linear-reward flexible-feature DGP",
    "shapeshifter_encoded_state_locally_robust": "encoded-state locally robust DGP",
    "shapeshifter_neural_neural": "raw-neural flexible diagnostic DGP",
    "tier4_neural_r_phi": "neural reward-and-feature stress DGP",
}

VALIDATION_ROLES: dict[str, str] = {
    "canonical_low_action": "low-dimensional check",
    "canonical_high_action": "high-dimensional diagnostic",
    "canonical_low_state_only": "state-only diagnostic",
    "canonical_latent_segments": "large diagnostic",
    "mce_low_high_reward": "primary validation",
    "airl_paper_identification": "paper-side validation",
    "airl_anchor_action_dependent": "anchored diagnostic",
    "airl_het_paper_identification": "paper-side validation",
    "f_irl_paper_state_marginal": "paper-side validation",
    "gladius_paper_high_state": "paper-side validation",
    "gladius_paper_high_state_scaled": "scaled diagnostic",
    "deep_mce_neural_reward": "primary validation",
    "deep_mce_neural_features": "projection diagnostic",
    "deep_mce_neural_reward_features": "stress test",
    "shapeshifter_linear_reward_neural_features": "finite-theta showcase",
    "shapeshifter_encoded_state_locally_robust": "paper-inference validation",
    "shapeshifter_neural_neural": "raw-neural diagnostic",
    "tier4_neural_r_phi": "neural stress test",
}

VALIDATION_CONTEXT_DEFINITIONS: dict[str, str] = {
    "canonical_low_action": (
        "A compact finite-state DDC benchmark with action-specific rewards, "
        "known transitions, known reward, value, policy, and Q-function truth, "
        "and Type A/B/C counterfactual oracles."
    ),
    "canonical_high_action": (
        "A harder encoded-state and richer reward-feature stress cell using "
        "the same known-truth reward, value, policy, Q-function, and Type A/B/C "
        "counterfactual objects to test scaling beyond the compact tabular case."
    ),
    "canonical_low_state_only": (
        "A diagnostic cell where rewards do not vary by action. It is used "
        "for estimators or identification arguments that are state-reward based."
    ),
    "canonical_latent_segments": (
        "A larger latent-heterogeneity diagnostic with segment-specific "
        "preferences and known segment-level truth."
    ),
    "mce_low_high_reward": (
        "A maximum-causal-entropy stress cell with a compact state space and "
        "a richer action-dependent reward-feature basis."
    ),
    "airl_paper_identification": (
        "A state-only, deterministic-transition AIRL cell that matches the "
        "original AIRL identification side conditions as closely as the shared "
        "harness allows."
    ),
    "airl_anchor_action_dependent": (
        "An anchored action-dependent AIRL boundary case. It checks the "
        "absorbing-state and exit-action gauge but is not the certified AIRL "
        "reward-recovery setting."
    ),
    "airl_het_paper_identification": (
        "A serialized-content heterogeneous AIRL cell with repeated users, two "
        "latent segments, pay/wait/exit actions, an exit-action reward anchor, "
        "and known segment-level truth."
    ),
    "f_irl_paper_state_marginal": (
        "A paper-side f-IRL cell with state-marginal matching and a state-only "
        "reward, included because that is the structural setting claimed by "
        "the estimator."
    ),
    "gladius_paper_high_state": (
        "A GLADIUS high-dimensional-state cell with low-dimensional reward "
        "features and an anchor-action Q loss, included to test projected "
        "structural reward recovery."
    ),
    "gladius_paper_high_state_scaled": (
        "A scaled GLADIUS diagnostic variant of the high-dimensional-state "
        "cell, used to probe whether the same gate failures persist under "
        "rescaling."
    ),
    "deep_mce_neural_reward": (
        "A Deep MCE-IRL cell with an anchored frozen neural reward map over "
        "supplied state encodings. The gated object is the recovered reward "
        "table, not neural-network weights."
    ),
    "deep_mce_neural_features": (
        "A Deep MCE-IRL projection diagnostic with neural state features and a "
        "finite projected reward target."
    ),
    "deep_mce_neural_reward_features": (
        "A Deep MCE-IRL stress cell with both nonlinear reward and neural "
        "feature structure, included to test reward-map recovery beyond the "
        "linear-feature sanity case."
    ),
    "shapeshifter_linear_reward_neural_features": (
        "A TD-CCP hard flexible cell with stochastic transitions, frozen neural "
        "state features, and a finite linear structural reward with action 0 "
        "normalized to zero."
    ),
    "shapeshifter_encoded_state_locally_robust": (
        "A TD-CCP hard cell with two-dimensional encoded state coordinates, "
        "stochastic transitions, logit first-stage CCPs, Algorithm 2 "
        "cross-fitting, locally robust standard errors, and a finite linear "
        "structural reward with action 0 normalized to zero."
    ),
    "shapeshifter_neural_neural": (
        "A raw-neural flexible diagnostic with a frozen neural reward matrix "
        "and no finite true theta, so it is not a finite-parameter recovery "
        "claim."
    ),
    "tier4_neural_r_phi": (
        "A neural reward-and-feature stress cell used to probe flexible-DGP "
        "behavior beyond the finite-theta validation surface."
    ),
}


def validation_display_name(cell_id: str) -> str:
    """Return a public label for an internal validation cell id."""

    return VALIDATION_DISPLAY_NAMES.get(cell_id, _fallback_label(cell_id))


def validation_role(cell_id: str, default: str = "validation") -> str:
    """Return a public role label for an internal validation cell id."""

    return VALIDATION_ROLES.get(cell_id, default)


def validation_context(cell_ids: list[str] | tuple[str, ...]) -> str:
    """Return a concise TeX explanation for the public validation DGP labels."""

    unique_cell_ids = list(dict.fromkeys(str(cell_id) for cell_id in cell_ids))
    if not unique_cell_ids:
        return (
            "This primer reports known-truth synthetic validation cells with "
            "machine-readable provenance kept outside the visible PDF prose."
        )

    if len(unique_cell_ids) == 1:
        cell_id = unique_cell_ids[0]
        return (
            rf"\textbf{{{_tex_text(validation_display_name(cell_id))}.}} "
            + _tex_text(_definition(cell_id))
        )

    lines = [r"\begin{itemize}"]
    for cell_id in unique_cell_ids:
        lines.append(
            rf"\item \textbf{{{_tex_text(validation_display_name(cell_id))}.}} "
            + _tex_text(_definition(cell_id))
        )
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def _fallback_label(cell_id: str) -> str:
    label = cell_id.replace("_", " ").strip()
    if not label:
        return "validation DGP"
    if "dgp" not in label.lower() and "diagnostic" not in label.lower():
        return f"{label} DGP"
    return label


def _definition(cell_id: str) -> str:
    return VALIDATION_CONTEXT_DEFINITIONS.get(
        cell_id,
        "A known-truth validation cell with generated reward, policy, value, "
        "and counterfactual objects used to interpret the estimator gates.",
    )


def _tex_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )
