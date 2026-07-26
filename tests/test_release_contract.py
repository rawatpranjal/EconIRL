"""Release metadata contracts."""

from __future__ import annotations

import runpy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import econirl

ROOT = Path(__file__).resolve().parents[1]


def test_package_versions_stay_synchronized() -> None:
    """Keep build metadata, the import surface, and Sphinx on one version."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    docs_config = runpy.run_path(str(ROOT / "docs" / "conf.py"))

    assert pyproject["project"]["version"] == econirl.__version__
    assert docs_config["release"] == econirl.__version__
