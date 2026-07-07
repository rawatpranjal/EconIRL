# Counterfactuals

Read this page as diagnostic counterfactual evidence. The regret numbers are
useful, but structural validity requires the upstream reward, value, Q, and
projection checks to pass.

GLADIUS projects implied rewards onto structural features via action-difference
regression. The projected parameters are only partially identified. The
action-difference projection removes a state-dependent additive constant. It
does not guarantee that the recovered reward matches the data-generating reward
in an absolute sense.

Full structural counterfactual validity requires both projected-reward recovery
and raw Bellman reward and value recovery. The current evaluation cells fail the
raw Bellman reward and value checks, so GLADIUS has not reached structural counterfactual validity.
Counterfactual analysis using projected GLADIUS parameters should be treated as
diagnostic, not as structural.

## Counterfactual Families Checked

The evaluation cells track welfare regret for three intervention types, compared
against oracle objects derived from the data-generating process:

| Family | Intervention | Checked against |
| --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | Oracle policy, value, and welfare regret. |
| Type B | Transition change (the dynamics change). | Oracle policy, value, and welfare regret. |
| Type C | Action removal (one action is penalized away). | Oracle policy, value, and welfare regret. |

## Reported Results

Counterfactual regret on the two synthetic cells:

| Cell | Type A regret | Type B regret | Type C regret |
| --- | ---: | ---: | ---: |
| `gladius_paper_high_state` | 0.00854 | 0.0529 | 0.00852 |
| `gladius_paper_high_state_scaled` | 0.00274 | 0.00742 | 0.000493 |

Regret is the welfare loss incurred by following the GLADIUS-projected policy
instead of the true optimal policy in the counterfactual world. The scaled cell
shows markedly lower regret, consistent with its better projected-reward
recovery.

These numbers are useful diagnostics but are not sufficient for structural
validity. Reaching structural counterfactual validity requires the full check
bundle - projected reward, raw Bellman reward, value, Q, and counterfactual
regret - to pass together. The raw Bellman reward and value checks
currently fail on both cells. See the [Simulation Study](validation.md) for the
full check table and the results files for exact metric values:
[gladius.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json)
and
[gladius_scaled.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_scaled.json).
