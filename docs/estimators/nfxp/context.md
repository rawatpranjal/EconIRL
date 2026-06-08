# Context

NFXP was introduced for dynamic discrete choice problems where agents choose
today while accounting for future states. Rust's bus-engine replacement model
is the standard example. A bus operator decides whether to keep an engine or
replace it. Keeping saves the replacement cost today but makes future mileage
states worse.

The estimator is useful because it connects observed choices to primitive
payoffs. Once the reward parameters and transition process are in hand, the
model can answer policy questions by solving the dynamic program again under a
changed payoff or environment.

## Source Ideas

The primary source is Rust's 1987 bus-engine replacement paper. The
computational upgrade used by EconIRL is the successive-approximation plus
Newton-Kantorovich inner-loop strategy associated with Iskhakov, Rust,
Schjerning, and Seo.

The core identification lesson is simple. Reward scale and location need a
normalization. Transitions need to be separated from payoffs. Reward features
need enough action variation to identify structural parameters.

## Where NFXP Fits

NFXP is the reference estimator for tabular structural estimation. CCP and
MPEC are useful comparisons because they target the same structural object with
different computational strategies. NNES, SEES, and TD-CCP become attractive
when exact nested Bellman solves are too expensive.
