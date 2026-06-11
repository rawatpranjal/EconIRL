# Bundled Datasets

Bundled datasets are package assets. They should be small, versionable, and
covered by loader tests.

Large raw source datasets should remain outside Git. If an example needs large
external data, the loader should document how to obtain it and should fail
cleanly when it is absent.

The Rust bus data is the reference bundled dataset for structural estimator
examples. Other bundled samples should be justified by tests, docs, or examples.
