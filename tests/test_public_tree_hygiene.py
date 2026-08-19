"""Keep maintainer-only material out of the tracked, public tree."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Absolute paths from a maintainer machine. Validation runners write the paths
# they actually used, so a re-run can reintroduce these into the receipts. The
# wheel and notebook checks build under a private temporary directory, so those
# roots leak the same way and belong in the same guard.
LOCAL_PATH = re.compile(
    r"(?:/(?:Users|home)/[^/\s\"']+/|/(?:private/)?(?:tmp|var/folders)/[^\s\"']+/)"
)

TEXT_SUFFIXES = {".py", ".json", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".ipynb", ".cff"}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / name for name in out.split("\0") if name]


def test_no_maintainer_paths_in_tracked_files() -> None:
    offenders = []
    for path in _tracked_files():
        if path.suffix not in TEXT_SUFFIXES or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in LOCAL_PATH.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {match.group(0)}")

    assert offenders == []


def test_agent_context_files_are_not_tracked() -> None:
    """Agent instruction files are maintainer-only, including suffixed forms."""

    offenders = [
        str(path.relative_to(ROOT))
        for path in _tracked_files()
        if path.name.endswith(("CLAUDE.md", "AGENTS.md")) or path.parts[-2:-1] == ("acceptance",)
    ]

    assert offenders == []
