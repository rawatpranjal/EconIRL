# Context

TD-CCP comes from Adusumilli and Eckardt's temporal-difference approach to
dynamic discrete choice estimation. It keeps the conditional choice probability
logic of Hotz-Miller style estimators, but estimates the recursive terms in
the pseudo-likelihood directly from observed state-action-successor tuples.

The practical point is to avoid specifying or estimating a transition density
while estimating the structural reward parameter. This matters when states are
continuous, high-dimensional, or represented through flexible encodings.

In that DDC class, the hard case is not "recover any neural reward matrix from
choices." The hard case is estimating a finite structural parameter
`theta` when the state space is awkward enough that transition-density
modeling, discretization, or repeated transition integration would be the
bottleneck. EconIRL's certified case therefore uses encoded continuous-style
state features, a finite reward index, stochastic transitions, and the paper's
locally robust cross-fitted correction.

## Source Ideas

The paper writes flow utility as a known feature vector times a
finite-dimensional parameter.

```text
u_theta(a, x) = z(a, x)' theta
```

The continuation terms in the CCP index are `h(a, x)' theta + g(a, x)`.
For Type I Extreme Value shocks, the expected shock correction is the Euler
constant minus the log conditional choice probability at `(a, x)`. TD-CCP
estimates `h` and `g` from observed `(a, x, a', x')` tuples rather than from a
transition-density model.

The linear semigradient path uses basis functions and projected TD normal
equations. The approximate value iteration path solves a sequence of
prediction problems and can use flexible learners. Algorithm 2 then adds
cross-fitting, a preliminary plug-in estimator, a backward `lambda` recursion,
and a locally robust `zeta` moment for paper-faithful inference.

## Where TD-CCP Fits

NFXP is the tabular reference because it solves the Bellman fixed point inside
the likelihood. CCP is fast when the first-stage policy and transition objects
are well supported. TD-CCP is useful when exact transition integration or
transition-density modeling is the main bottleneck.

TD-CCP still targets structural reward parameters. Behavioral cloning stops at
the observed policy. Raw neural reward recovery is a different target and is
not the certified TD-CCP claim in EconIRL.

## What RTD Does Not Claim

| Boundary | Reason |
| --- | --- |
| Unrestricted raw neural reward recovery | The paper's structural target is finite `theta` in known reward features. |
| Transition-free counterfactual evaluation | Counterfactual policy/value evaluation still needs a transition environment. |
| Certified neural AVI inference | AVI is available, but the current certified artifact is the semigradient Algorithm 2 path. |
| Replacement for tabular NFXP when transitions are known and small | TD-CCP is most useful when transition-density modeling is the bottleneck. |
