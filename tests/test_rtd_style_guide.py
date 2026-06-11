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
        "algorithm sketch": re.compile(
            r"\balgorithm sketch\b", flags=re.IGNORECASE
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


def test_estimator_navigation_is_owned_by_estimators_page() -> None:
    """Keep estimator links under the Estimators section, not hardcoded at root."""

    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    estimator_overview = (DOCS / "estimators.md").read_text(encoding="utf-8")
    api_design = (DOCS / "user_guide" / "api_design.md").read_text(
        encoding="utf-8"
    )
    config = runpy.run_path(str(DOCS / "conf.py"))
    expected = [
        "estimators/nfxp",
        "estimators/ccp",
        "estimators/mpec",
        "estimators/nnes",
        "estimators/tdccp",
        "estimators/mce_irl",
        "estimators/deep_mce_irl",
        "estimators/airl",
        "estimators/airl_het",
        "estimators/f_irl",
        "estimators/gladius",
        "estimators/iq_learn",
    ]

    root_entries = [entry for entry in expected if f"   {entry}\n" in index]
    estimator_entries = [
        entry for entry in expected if f"{entry}\n" not in estimator_overview
    ]
    assert root_entries == []
    assert estimator_entries == []
    assert config["html_theme_options"]["navigation_depth"] == 2
    assert "Estimator Map" not in index
    assert "Estimator Map" not in estimator_overview
    assert "Structural Econometrics" not in estimator_overview
    assert "Inverse Reinforcement Learning" not in estimator_overview
    assert "```{toctree}" in estimator_overview
    assert "   references\n" in index
    assert api_design.startswith("# API Design\n")
    assert "Problem Setup and API Design" not in api_design


def test_estimator_landing_pages_do_not_expand_sidebar_guides() -> None:
    """Keep per-estimator guide pages out of the RTD sidebar tree."""

    pages = [
        page
        for page in sorted((DOCS / "estimators").glob("*.md"))
        if not _is_excluded_from_rtd(page)
    ]
    assert pages

    offenders = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for block in _toctree_blocks(text):
            if ":hidden:" not in block:
                offenders.append(
                    f"{page.relative_to(ROOT)}: estimator guide toctree must be hidden"
                )
                continue
            for entry in _toctree_entries(block):
                if f"]({entry}.md)" not in text:
                    offenders.append(
                        f"{page.relative_to(ROOT)}: visible guide link missing for {entry}"
                    )

    assert offenders == []


def test_estimator_pages_name_source_papers_up_front() -> None:
    """Keep estimator pages explicit about the papers they draw from."""

    pages = [
        page
        for page in sorted((DOCS / "estimators").glob("*.md"))
        if not _is_excluded_from_rtd(page)
    ]
    assert pages

    first_content_markers = [
        "## Quick Decision",
        "## When to Use",
        "## When To Use It",
        "## Basic Usage",
        "## Questions NNES Answers",
        "## Simulation Study",
        "## Evidence",
    ]

    offenders = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        source_pos = text.find("## Source Papers")
        if source_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Source Papers")
            continue

        first_marker_pos = min(
            (
                pos
                for marker in first_content_markers
                if (pos := text.find(marker)) != -1
            ),
            default=-1,
        )
        if first_marker_pos != -1 and source_pos > first_marker_pos:
            offenders.append(
                f"{page.relative_to(ROOT)}: Source Papers must be near the top"
            )

        source_section = _section_body(text, source_pos)
        if "{ref}`" not in source_section:
            offenders.append(
                f"{page.relative_to(ROOT)}: Source Papers must link to references"
            )

    assert offenders == []


def test_under_the_hood_pages_order_model_before_pseudocode() -> None:
    """Keep internals pages in setup, model, pseudocode order."""

    pages = sorted((DOCS / "estimators").glob("*/under_the_hood.md"))
    assert pages

    offenders = []
    pseudocode_block = re.compile(
        r"## Pseudocode\s+```text\s+[\s\S]+?\s+```", flags=re.IGNORECASE
    )
    for page in pages:
        text = page.read_text(encoding="utf-8")
        setup_pos = text.find("## Optimization Setup")
        model_match = re.search(
            r"^## Model(?: Objects)?\s*$", text, flags=re.MULTILINE
        )
        model_pos = model_match.start() if model_match else -1
        pseudocode_pos = text.find("## Pseudocode")
        if setup_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Optimization Setup")
            continue
        if model_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Model section")
            continue
        if pseudocode_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Pseudocode")
            continue
        if not setup_pos < model_pos < pseudocode_pos:
            offenders.append(
                f"{page.relative_to(ROOT)}: expected setup, model, pseudocode order"
            )
        if not pseudocode_block.search(text):
            offenders.append(f"{page.relative_to(ROOT)}: missing text pseudocode block")

    assert offenders == []


def test_references_page_is_in_public_navigation() -> None:
    """Keep source-paper citations reachable from RTD."""

    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    references = (DOCS / "references.md").read_text(encoding="utf-8")
    expected_ids = [
        "rust-1987",
        "hotz-miller-1993",
        "aguirregabiria-mira-2002",
        "su-judd-2012",
        "iskhakov-2016",
        "luo-sang-2024",
        "nguyen-2025",
        "adusumilli-eckardt-2025",
        "ziebart-2008",
        "ziebart-2010",
        "wulfmeier-2015",
        "fu-2018",
        "lee-sudhir-wang-2026",
        "ni-2020",
        "garg-2021",
        "kang-2025",
        "kim-2021",
        "cao-2021",
    ]

    missing_ids = [
        reference_id
        for reference_id in expected_ids
        if f"({reference_id})=" not in references
    ]

    assert "   references\n" in index
    assert references.startswith("# References\n")
    assert missing_ids == []


def _public_doc_sources() -> list[Path]:
    roots = [
        DOCS / "index.rst",
        DOCS / "estimators.md",
        DOCS / "references.md",
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


def _section_body(text: str, heading_pos: int) -> str:
    body_start = text.find("\n", heading_pos)
    if body_start == -1:
        return ""
    next_heading = text.find("\n## ", body_start + 1)
    if next_heading == -1:
        return text[body_start:]
    return text[body_start:next_heading]


def _toctree_blocks(text: str) -> list[str]:
    return re.findall(r"```{toctree}\n([\s\S]+?)\n```", text)


def _toctree_entries(block: str) -> list[str]:
    return [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith(":")
    ]
