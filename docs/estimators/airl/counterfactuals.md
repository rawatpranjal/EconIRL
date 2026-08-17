# Counterfactuals

## Important Links

- [AIRL Overview](../airl.md)
- [Identification Boundary](identification.md)
- [Taxi Dynamics Transfer](taxi_transfer.md)
- [Applied Notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/airl/airl_applied_workflow.ipynb)

AIRL supports counterfactuals that preserve the state-only reward
interpretation. The fitted policy is re-solved after one primitive changes.

## Supported changes

| Change | Public call |
| --- | --- |
| Transition system | `counterfactual(transitions=...)` |
| Reward parameters | `counterfactual(params=...)` |

Supply exactly one change. The result contains baseline and changed policies,
values, policy differences, and value differences.

A transition change holds the recovered reward fixed and re-solves behavior.
A parameter change is a user-specified reward scenario in the fitted basis. It
is not an identified causal effect.

The transition tensor retains `(n_actions, n_states, n_states)` orientation.
The method checks the tensor shape before solving. Supply a finite, nonnegative
tensor whose rows sum to one.

## Interpretation

A changed-dynamics result carries the AIRL transfer interpretation only when
the reward is state-only and the problem satisfies the decomposability
conditions. A small policy distance under the original dynamics is not enough.

The controlled study reports changed-dynamics policy TV of 0.0101 and regret of
0.0070 at the 95th percentile. The taxi study uses a larger intervention. Its
oracle policy changes by 0.1095 TV. AIRL reaches 0.0525 transfer policy TV and
0.0071 flow-equivalent regret at the 95th percentile.

## Unsupported changes

The public AIRL class does not accept action-dependent reward features,
context, or latent segments. It also does not expose action removal as a public
counterfactual input. Do not encode these changes by silently altering the
state feature basis.

Bootstrap intervals describe fitted reward and policy functionals. They are not
automatically propagated through a counterfactual. Report counterfactual
sampling uncertainty only after defining and validating that additional
resampling procedure.
