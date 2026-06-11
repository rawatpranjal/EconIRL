# Transition Tensor Conventions

Estimator-facing transition tensors commonly use
`(n_actions, n_states, n_states)`, where the axes are action, current state,
and next state.

Some internal code uses `(n_states, n_actions, n_states)`. Any boundary that
accepts transitions should document the expected orientation.

Rows should be stochastic. Validation code should check row sums and surface
the maximum transition row error.
