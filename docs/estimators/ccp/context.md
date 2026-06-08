# Context

CCP was introduced for dynamic discrete choice problems where researchers can
estimate the observed policy directly from data. Hotz and Miller showed that
conditional choice probabilities invert the dynamic choice problem under the
standard additive logit shock assumptions.

The practical appeal is speed. NFXP solves the Bellman equation inside every
likelihood evaluation. CCP estimates a policy first, converts that policy into
continuation-value terms, and then solves a standard logit likelihood over
augmented features.

## Source Ideas

The primary source is Hotz and Miller's 1993 conditional choice probability
paper. Aguirregabiria and Mira's nested pseudo-likelihood algorithm adds the
iteration that updates the policy after each pseudo-likelihood fit.

The core identification lesson matches NFXP. Reward scale and location need a
normalization. Transitions need to be separated from payoffs. Reward features
need enough action variation to identify structural parameters.

CCP adds one more requirement. The first-stage policy must have support across
the relevant actions. Sparse empirical CCPs make the inversion noisy.

## Where CCP Fits

CCP targets the same structural reward object as NFXP in finite tabular
dynamic discrete choice models. It is a useful production estimator when the
first-stage policy is well supported, and it is a useful comparison estimator
when NFXP is the reference.

TD-CCP and NNES become attractive when exact matrix inversion or exact Bellman
solves are too costly. Behavioral cloning is a lower bound because it stops at
the first-stage policy and does not recover structural rewards.
