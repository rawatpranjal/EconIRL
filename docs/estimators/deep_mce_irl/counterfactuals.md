# Counterfactuals

Deep MCE-IRL recovers an anchored reward matrix rather than a structural
parameter vector, so counterfactual analysis works differently from the
structural family. The reward matrix is re-solved under the perturbed
environment using the same soft Bellman operator, and the new policy and value
function are compared against the oracle counterfactual objects.

## Reward Transfer Mechanics

The three counterfactual families tested in the validation suite are:

| Family | Intervention | Transfer method |
| --- | --- | --- |
| Type A | Reward shift (a payoff component changes). | Modify the anchored reward matrix and re-solve. |
| Type B | Transition change (the dynamics change). | Use the same reward matrix with the new transition tensor. |
| Type C | Action removal (one action is penalized away). | Add a large penalty to the removed action's reward column and re-solve. |

For each family the welfare regret is the initial-distribution-weighted gap
between the oracle value and the value under the transferred policy:

$$
\text{regret} = \rho_0 \cdot (V^* - V^{\text{transfer}}).
$$

## Reported Results

On the primary synthetic cell the welfare regret of Deep MCE-IRL's
counterfactual policies is below 0.002 in every family (Type A 0.00164,
Type B 0.00148, Type C 0.00191; see the
[results file](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl.json)
and the [Simulation Study](validation.md) page). The learned reward is defined
on the same discrete state-action space as the oracle and uses the same
anchor, so it re-solves cleanly under all three intervention types.

## Limitations

The neural reward map is anchored but not structurally identified. Across
environments with different transition dynamics or state spaces, the reward
values are not directly comparable in the way structural parameters are. Use
counterfactual re-solving only when the environment being perturbed shares the
same state space and anchor as the estimation environment.

Raw neural network weights should not be transferred or compared across
separate fits. The reward matrix output by `model.reward_` is the correct
object to carry into a counterfactual solve.
