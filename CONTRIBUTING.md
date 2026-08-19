# Contributing to EconIRL

EconIRL is research software for structural dynamic discrete choice and inverse
reinforcement learning. Contributions are welcome when they preserve the
package's validation-first standard: public claims should trace to tests,
examples, source papers, or machine-readable validation artifacts.

## Ways to Contribute

- Report bugs or reproducibility failures through GitHub issues.
- Improve documentation, examples, or estimator explanations.
- Add tests for existing estimators, especially standard errors,
  counterfactuals, and validation scripts.
- Propose new estimators only with a clear evidence plan and scoped validation
  target.

## Development Setup

```bash
git clone https://github.com/rawatpranjal/econirl.git
cd econirl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Local Checks

Run focused tests before opening a pull request:

```bash
python -m pytest tests/ -v -m "not slow"
```

For estimator or validation changes, also run the relevant slow or validation
tests and record the command in the pull request. Do not update validation
numbers by hand without regenerating the corresponding artifact.

## Pull Requests

Please include:

- The problem being solved.
- The files and APIs changed.
- The tests or validation scripts run.
- Any limitations, numerical tolerances, or evidence gaps.

Large estimator additions should start as an issue or draft pull request with a
validation plan. The preferred bar is a source-paper replication when public
data and designs are available; otherwise use a known-truth or paper-design
check and label it clearly.

## Support

Use GitHub issues for bug reports, installation failures, documentation gaps,
or questions about validation evidence. Please include the package version,
Python version, operating system, and a minimal reproducible example when
possible.
