# Counterfactuals

AIRL does not provide a structural gauge. The recovered reward component
$g_\theta$ is identified only up to potential-based shaping, so raw parameter
values cannot be re-solved under an arbitrary intervention the way NFXP or
MCE-IRL can. The package evaluates AIRL counterfactuals through the simulation
harness, which fixes the intervention oracle-side and measures the welfare cost
of deploying the recovered policy in the changed world.

## Counterfactual Families

| Type | Intervention | How AIRL is evaluated |
| --- | --- | --- |
| Type A | Shift payoffs, hold transitions fixed. | Re-solve the oracle problem under the new payoffs; report regret from the old AIRL policy. |
| Type B | Change transitions, hold the recovered reward fixed. | Re-solve the oracle under new transitions; report regret from the old AIRL policy. |
| Type C | Disable one non-anchor action. | Re-solve the oracle with the penalized action; report regret from the old AIRL policy. |

Because AIRL's recovered reward is state-only and not anchored to a structural
gauge, it is not re-solved under each intervention. The harness instead treats
the recovered policy as frozen and measures how much welfare it loses in the
counterfactual world relative to the oracle policy that was re-solved under the
intervention.

## Reported Results

From the primary `airl_paper_identification` simulation:

| Counterfactual | Policy TV | Regret |
| --- | ---: | ---: |
| Type A | 0.0062 | 0.0029 |
| Type B | 0.0076 | 0.0038 |
| Type C | 0.0075 | 0.0050 |

Regret values below 0.01 are in the range of good IRL methods on this synthetic
cell. They reflect a nearly correct policy in the base world; the recovered
policy adapts reasonably to the counterfactual world even without re-solving,
because the oracle policy change is also small.

## API Boundary

The stable public objects after fitting are `summary.policy`, `summary.parameters`,
and the value function accessible from the lower-level result. For controlled
payoff, transition, or action-set interventions, use the package's simulation
and evaluation utilities with an explicit problem and transition environment.
The `AIRLEstimator` does not expose a `.counterfactual()` method; the
counterfactual evaluation in the results file comes from the validation harness.
