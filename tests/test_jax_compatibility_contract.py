"""Source-level contracts for supported JAX versions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_jax_clip_avoids_removed_one_sided_keywords() -> None:
    """Keep source and tests compatible with current JAX."""

    offenders: list[str] = []
    for root in (ROOT / "src", ROOT / "tests"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "clip"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "jnp"
                ):
                    continue
                for keyword in node.keywords:
                    if keyword.arg in {"a_min", "a_max"}:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {keyword.arg}")

    assert offenders == []
