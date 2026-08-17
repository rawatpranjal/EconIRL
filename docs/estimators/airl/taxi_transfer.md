# Taxi Dynamics Transfer

## Important Links

- [AIRL Overview](../airl.md)
- [Counterfactuals](counterfactuals.md)
- [Taxi transfer results](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/airl_taxi_transfer.json)
- [Applied Notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/airl/airl_applied_workflow.ipynb)

This generated study asks whether a taxi-zone reward remains useful after
traffic dynamics change.

## Design

The city is a 4 by 4 grid with 16 zones and four movement actions. Zone reward
depends on downtown access, airport access, and a congestion-zone indicator.
All three features are state-only.

Training routes use deterministic movement. The changed environment lowers the
reliability of northbound and eastbound movement. It also closes an eastbound
corridor into downtown. The reward is held fixed while the transition tensor
changes.

Each of three replications uses 300 taxis observed for 80 periods. State-action
coverage ranges from 0.9531 to 1.0000. The state feature matrix has rank 3.

## Results

| Metric | Median | 95th percentile |
| --- | ---: | ---: |
| Reward normalized RMSE | 0.0689 | 0.0897 |
| Training policy TV | 0.0402 | 0.0498 |
| Transfer policy TV | 0.0426 | 0.0525 |
| Discount-normalized transfer regret (lower is better) | 0.0047 | 0.0071 |
| AIRL policy change TV | 0.1039 | 0.1045 |

The oracle policy change is 0.1095 TV, so the changed dynamics have a material
effect on behavior. AIRL changes its policy by a similar amount after re-solving
the recovered reward.

Zone 9 has the largest fitted policy change in every replication. In replication
1, eastbound probability falls from 0.2931 to 0.0900. Southbound
probability rises from 0.2069 to 0.4340. The model routes taxis around the
closure instead of deploying the training policy unchanged.

## Scope

This is a generated application study. It is not a study from Fu et al. The
result supports changed-dynamics re-optimization for this state-only design. It
does not support action-dependent taxi payoffs, latent driver segments, or a
claim that every grid satisfies the paper's decomposability condition.
