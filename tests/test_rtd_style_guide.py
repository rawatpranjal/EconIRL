"""Source-level checks for RTD evidence prose."""

from __future__ import annotations

import re
import runpy
from fnmatch import fnmatch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_estimator_validation_pages_render_as_simulation_studies() -> None:
    """Keep the stable validation.md path but not the reader-facing label."""

    pages = [
        page
        for page in sorted((DOCS / "estimators").glob("*/validation.md"))
        if not _is_excluded_from_rtd(page)
    ]
    assert pages

    offenders = []
    for page in pages:
        first_heading = next(
            (
                line.strip()
                for line in page.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("#")
            ),
            "",
        )
        if first_heading != "# Simulation Study":
            offenders.append(f"{page.relative_to(ROOT)}: {first_heading}")

    assert offenders == []


def test_estimator_simulation_study_pages_have_study_openings() -> None:
    """Lock the checklist from the JSS/RTD simulation-study rule."""

    required_patterns = {
        "study question": re.compile(r"\bsimulation\s+asks\b", flags=re.IGNORECASE),
        "real-data boundary": re.compile(
            r"\breal[\s\S]{0,80}data cannot answer\b", flags=re.IGNORECASE
        ),
        "generated panel": re.compile(r"\b(panel|demonstrations)\b"),
        "estimator sees": re.compile(r"\bestimator sees\b", flags=re.IGNORECASE),
        "truth held back": re.compile(
            r"\bheld\s+back\s+for\s+evaluation\b", flags=re.IGNORECASE
        ),
    }

    offenders = []
    for page in _public_estimator_validation_pages():
        intro = _intro_before_first_subheading(page)
        for name, pattern in required_patterns.items():
            if not pattern.search(intro):
                offenders.append(f"{page.relative_to(ROOT)}: missing {name}")

    assert offenders == []


def test_public_rtd_source_avoids_release_claim_wording() -> None:
    """Catch stale release/certification wording before public docs pushes."""

    patterns = {
        "known truth": re.compile(r"\bknown[-_ ]truth\b", flags=re.IGNORECASE),
        "threshold check families": re.compile(
            r"\bthreshold check families\b", flags=re.IGNORECASE
        ),
        "threshold checks": re.compile(
            r"\bthreshold checks\b", flags=re.IGNORECASE
        ),
        "scope labels": re.compile(r"\bscope labels\b", flags=re.IGNORECASE),
        "artifact": re.compile(r"\bartifacts?\b", flags=re.IGNORECASE),
        "certified as": re.compile(r"\bcertified as\b", flags=re.IGNORECASE),
        "release claim": re.compile(r"\brelease claim\b", flags=re.IGNORECASE),
        "validation target": re.compile(
            r"\bvalidation target\b", flags=re.IGNORECASE
        ),
    }

    offenders = []
    for path in _public_doc_sources():
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {name}")

    assert offenders == []


def test_estimator_docs_use_simulation_study_links_and_terms() -> None:
    """Avoid drifting back to validation-page prose on estimator RTD pages."""

    patterns = {
        "Validation link label": re.compile(r"\[Validation\]\(validation\.md\)"),
        "validation harness": re.compile(
            r"\bvalidation harness\b", flags=re.IGNORECASE
        ),
        "validation evidence": re.compile(
            r"\bvalidation evidence\b", flags=re.IGNORECASE
        ),
        "validation objects": re.compile(
            r"\bvalidation objects\b", flags=re.IGNORECASE
        ),
        "validation page reports": re.compile(
            r"\bvalidation page reports\b", flags=re.IGNORECASE
        ),
        "validation surface": re.compile(
            r"\bvalidation surface\b", flags=re.IGNORECASE
        ),
        "known-truth validation": re.compile(
            r"\bknown-truth validation\b", flags=re.IGNORECASE
        ),
    }

    offenders = []
    for path in _public_estimator_sources():
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {name}")

    assert offenders == []


def test_under_the_hood_pages_start_with_optimization_and_pseudocode() -> None:
    """Keep estimator internals pages algorithm-first."""

    pages = sorted((DOCS / "estimators").glob("*/under_the_hood.md"))
    assert pages

    offenders = []
    pseudocode_block = re.compile(
        r"## Pseudocode\s+```text\s+[\s\S]+?\s+```", flags=re.IGNORECASE
    )
    for page in pages:
        text = page.read_text(encoding="utf-8")
        setup_pos = text.find("## Optimization Setup")
        pseudocode_pos = text.find("## Pseudocode")
        if setup_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Optimization Setup")
            continue
        if pseudocode_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Pseudocode")
            continue
        if setup_pos > pseudocode_pos:
            offenders.append(f"{page.relative_to(ROOT)}: setup must precede pseudocode")
        if not pseudocode_block.search(text):
            offenders.append(f"{page.relative_to(ROOT)}: missing text pseudocode block")

    assert offenders == []


def _public_doc_sources() -> list[Path]:
    roots = [
        DOCS / "index.rst",
        DOCS / "estimators.md",
        DOCS / "estimators",
        DOCS / "user_guide",
    ]
    sources: list[Path] = []
    for root in roots:
        if root.is_file():
            sources.append(root)
            continue
        sources.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.suffix in {".md", ".rst"} and not _is_excluded_from_rtd(path)
        )
    return sources


def _public_estimator_sources() -> list[Path]:
    return [
        path
        for path in sorted((DOCS / "estimators").rglob("*.md"))
        if not _is_excluded_from_rtd(path)
    ]


def _public_estimator_validation_pages() -> list[Path]:
    return [
        path
        for path in sorted((DOCS / "estimators").glob("*/validation.md"))
        if not _is_excluded_from_rtd(path)
    ]


def _is_excluded_from_rtd(path: Path) -> bool:
    rel = path.relative_to(DOCS).as_posix()
    parts = path.relative_to(DOCS).parts
    for pattern in _exclude_patterns():
        if pattern in {"_build", "archive"} and pattern in parts:
            return True
        if fnmatch(rel, pattern):
            return True
        if pattern.endswith("/**") and (
            rel == pattern[:-3] or rel.startswith(pattern[:-2])
        ):
            return True
    return False


def _exclude_patterns() -> list[str]:
    config = runpy.run_path(str(DOCS / "conf.py"))
    return list(config["exclude_patterns"])


def _intro_before_first_subheading(page: Path) -> str:
    lines = page.read_text(encoding="utf-8").splitlines()
    intro: list[str] = []
    saw_title = False
    for line in lines:
        if line.startswith("# "):
            saw_title = True
            continue
        if saw_title and line.startswith("## "):
            break
        if saw_title:
            intro.append(line)
    return "\n".join(intro)
