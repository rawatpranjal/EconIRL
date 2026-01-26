# Changelog

All notable changes to econirl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **MCE IRL parameter recovery**: Fixed critical bug where expected features were computed using stationary distribution instead of empirical state sequence. This caused gradients to be non-zero at true parameters, preventing convergence. The fix iterates over the empirical state sequence from demonstrations, reducing gradient norm at true parameters by 25-250x. ([Kim et al. 2021](https://proceedings.mlr.press/v139/kim21c.html) provides theoretical grounding for reward identifiability in IRL.)

### Added
- **tqdm progress tracking** in MCE IRL with optional RMSE display when `true_params` is provided for debugging
- **`true_params` argument** to `MCEIRLEstimator.estimate()` for monitoring parameter recovery during optimization

### Changed
- `_compute_expected_features()` signature now takes `panel` instead of `state_visitation` to enable iteration over empirical states

## [0.1.0] - 2025-01-24

### Added
- Initial release with NFXP, CCP, MCE IRL, and MaxEnt IRL estimators
- Rust (1987) bus engine replacement environment
- Action-dependent reward functions for structural models
- Bootstrap and asymptotic standard error computation
- Visualization tools for policies and value functions
