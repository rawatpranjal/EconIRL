"""Source-level checks for RTD evidence prose."""

from __future__ import annotations

import re
import runpy
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NFXP_PAGES = [
    DOCS / "estimators" / "nfxp.md",
    DOCS / "estimators" / "nfxp" / "quick_start.md",
    DOCS / "estimators" / "nfxp" / "pre_estimation.md",
    DOCS / "estimators" / "nfxp" / "validation.md",
    DOCS / "estimators" / "nfxp" / "counterfactuals.md",
    DOCS / "estimators" / "nfxp" / "rust_bus.md",
]
CCP_PAGES = [
    DOCS / "estimators" / "ccp.md",
    DOCS / "estimators" / "ccp" / "quick_start.md",
    DOCS / "estimators" / "ccp" / "pre_estimation.md",
    DOCS / "estimators" / "ccp" / "validation.md",
    DOCS / "estimators" / "ccp" / "counterfactuals.md",
    DOCS / "estimators" / "ccp" / "rust_bus.md",
]
NEURAL_MCE_PAGES = [
    DOCS / "estimators" / "deep_mce_irl.md",
    DOCS / "estimators" / "deep_mce_irl" / "quick_start.md",
    DOCS / "estimators" / "deep_mce_irl" / "pre_estimation.md",
    DOCS / "estimators" / "deep_mce_irl" / "validation.md",
    DOCS / "estimators" / "deep_mce_irl" / "counterfactuals.md",
    DOCS / "estimators" / "deep_mce_irl" / "wulfmeier_objectworld.md",
]
AIRL2_PAGES = [
    DOCS / "estimators" / "airl2.md",
    DOCS / "estimators" / "airl2" / "quick_start.md",
    DOCS / "estimators" / "airl2" / "pre_estimation.md",
    DOCS / "estimators" / "airl2" / "validation.md",
    DOCS / "estimators" / "airl2" / "counterfactuals.md",
    DOCS / "estimators" / "airl2" / "serialized_content.md",
]
COMPLETED_ESTIMATOR_PAGES = NFXP_PAGES + CCP_PAGES + NEURAL_MCE_PAGES + AIRL2_PAGES


def test_completed_estimator_pages_put_important_links_immediately_after_title() -> None:
    """Keep the most useful destinations visible at the top."""

    pattern = re.compile(
        r"\A# [^\n]+\n\n"
        r"## Important Links\n\n"
        r"(?P<links>(?:- \[[^\]]+\]\([^)]+\)\n){3,5})"
        r"\n"
    )
    offenders = [
        str(page.relative_to(ROOT))
        for page in COMPLETED_ESTIMATOR_PAGES
        if pattern.match(page.read_text(encoding="utf-8")) is None
    ]

    assert offenders == []


def test_completed_estimator_executable_snippets_show_exact_results() -> None:
    """Require a non-empty result block after every Python or shell example."""

    offenders = []
    executable_blocks = 0

    for page in COMPLETED_ESTIMATOR_PAGES:
        lines = page.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            if lines[index].strip() not in {"```python", "```bash"}:
                index += 1
                continue

            executable_blocks += 1
            snippet_line = index + 1
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                index += 1
            if index == len(lines):
                offenders.append(f"{page.relative_to(ROOT)}:{snippet_line}: unclosed snippet")
                break

            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index == len(lines) or lines[index].strip() != "**Result**":
                offenders.append(f"{page.relative_to(ROOT)}:{snippet_line}: missing Result label")
                continue

            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index == len(lines) or lines[index].strip() != "```text":
                offenders.append(f"{page.relative_to(ROOT)}:{snippet_line}: missing text result")
                continue

            index += 1
            result_lines = []
            while index < len(lines) and lines[index].strip() != "```":
                result_lines.append(lines[index])
                index += 1
            if index == len(lines):
                offenders.append(f"{page.relative_to(ROOT)}:{snippet_line}: unclosed result")
                break
            result = "\n".join(result_lines).strip()
            if not result:
                offenders.append(f"{page.relative_to(ROOT)}:{snippet_line}: empty result")
            if "..." in result or "…" in result:
                offenders.append(f"{page.relative_to(ROOT)}:{snippet_line}: abbreviated result")

    assert executable_blocks
    assert offenders == []


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
    """Lock the invariants of the 2026-06-12 simulation-study page restyle.

    The old style required literal phrases ("simulation asks", "real data
    cannot answer", "estimator sees", "held back for evaluation") in the intro
    before the first subheading.  The June 2026 restyle dropped those exact
    phrases but kept two concrete requirements that are verifiable and
    meaningful:

    1. A link to ``validation/results/`` — the result file that backs every
       numerical claim.
    2. A ``PYTHONPATH=src:.`` reproduce command — so the result can be re-run
       from source.

    A page that omits either of these provides numerical claims with no
    traceable evidence or no way to reproduce them, which violates the
    simulation-study contract.
    """

    required_patterns = {
        "result file link": re.compile(r"validation/results/", flags=re.IGNORECASE),
        "reproduce command": re.compile(r"PYTHONPATH=src:\.", flags=re.IGNORECASE),
    }

    offenders = []
    for page in _public_estimator_validation_pages():
        text = page.read_text(encoding="utf-8")
        for name, pattern in required_patterns.items():
            if not pattern.search(text):
                offenders.append(f"{page.relative_to(ROOT)}: missing {name}")

    assert offenders == []


def test_public_rtd_source_avoids_release_claim_wording() -> None:
    """Catch stale release/certification wording before public docs pushes."""

    patterns = {
        "known truth": re.compile(r"\bknown[-_ ]truth\b", flags=re.IGNORECASE),
        "threshold check families": re.compile(
            r"\bthreshold check families\b", flags=re.IGNORECASE
        ),
        "threshold checks": re.compile(r"\bthreshold checks\b", flags=re.IGNORECASE),
        "scope labels": re.compile(r"\bscope labels\b", flags=re.IGNORECASE),
        "artifact": re.compile(r"\bartifacts?\b", flags=re.IGNORECASE),
        "certified as": re.compile(r"\bcertified as\b", flags=re.IGNORECASE),
        "release claim": re.compile(r"\brelease claim\b", flags=re.IGNORECASE),
        "validation target": re.compile(r"\bvalidation target\b", flags=re.IGNORECASE),
        "algorithm sketch": re.compile(r"\balgorithm sketch\b", flags=re.IGNORECASE),
        # Register banned 2026-06-12 (docs/research/internal_docs/style.md, public prose
        # register): internal honesty-contract vocabulary that leaked onto
        # the live RTD pages.
        "gauge": re.compile(r"\bgauges?\b", flags=re.IGNORECASE),
        "verbatim": re.compile(r"\bverbatim\b", flags=re.IGNORECASE),
        "honest": re.compile(r"\bhonest(?:ly|y)?\b", flags=re.IGNORECASE),
        "frozen policy": re.compile(r"\bfrozen\b", flags=re.IGNORECASE),
        "teaching arc": re.compile(r"\bteaching arc\b", flags=re.IGNORECASE),
        "through-line": re.compile(r"\bthrough-?line\b", flags=re.IGNORECASE),
        # Style-guide bans not previously enforced (roadmap D1, 2026-06-15).
        # \s+ (not a literal space) so a phrase split across a line break is
        # still caught — Markdown collapses the newline when rendering.
        "machine-readable": re.compile(r"\bmachine[-\s]readable\b", flags=re.IGNORECASE),
        "convergence flag": re.compile(r"\bconverg(?:ed|ence)\s+flag\b", flags=re.IGNORECASE),
        "summary exposes": re.compile(r"\bsummary\s+exposes\b", flags=re.IGNORECASE),
        "fitted summary reports": re.compile(
            r"\bfitted\s+summary\s+reports\b", flags=re.IGNORECASE
        ),
        "evidence scope": re.compile(r"\bevidence\s+scope\b", flags=re.IGNORECASE),
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
        "validation harness": re.compile(r"\bvalidation harness\b", flags=re.IGNORECASE),
        "validation evidence": re.compile(r"\bvalidation evidence\b", flags=re.IGNORECASE),
        "validation objects": re.compile(r"\bvalidation objects\b", flags=re.IGNORECASE),
        "validation page reports": re.compile(r"\bvalidation page reports\b", flags=re.IGNORECASE),
        "validation surface": re.compile(r"\bvalidation surface\b", flags=re.IGNORECASE),
        "known-truth validation": re.compile(r"\bknown-truth validation\b", flags=re.IGNORECASE),
    }

    offenders = []
    for path in _public_estimator_sources():
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {name}")

    assert offenders == []


def test_estimator_navigation_is_owned_by_section_pages() -> None:
    """Estimator links live in exactly one roster section, not at the root."""

    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    core = (DOCS / "estimators" / "core.md").read_text(encoding="utf-8")
    other = (DOCS / "estimators" / "other.md").read_text(encoding="utf-8")
    config = runpy.run_path(str(DOCS / "conf.py"))

    # The core roster lives in core.md; other estimators live in other.md.
    expected_core = [
        "nfxp",
        "ccp",
        "mce_irl",
        "deep_mce_irl",
        "airl",
        "neural_airl",
        "gladius",
    ]
    expected_other = [
        "tdccp",
        "airl2",
        "nnes",
        "mpec",
        "ufxp",
        "rhip",
        "f_irl",
        "iq_learn",
    ]
    missing_core = [entry for entry in expected_core if f"\n{entry}\n" not in core]
    missing_other = [entry for entry in expected_other if f"\n{entry}\n" not in other]
    other_in_core = [entry for entry in expected_other if f"\n{entry}\n" in core]
    assert missing_core == []
    assert missing_other == []
    assert other_in_core == []

    # Estimator pages are not hardcoded directly in the root toctree.
    root_entries = [
        entry for entry in (expected_core + expected_other) if f"   estimators/{entry}\n" in index
    ]
    assert root_entries == []

    # The two section pages are the top-level estimator entries.
    assert "   estimators/core\n" in index
    assert "   estimators/other\n" in index

    assert config["html_theme_options"]["navigation_depth"] == 2
    assert "Estimator Map" not in index
    assert "```{toctree}" in core
    assert "```{toctree}" in other
    assert "   references\n" in index
    assert "   user_guide/your_own_data\n" in index


def test_estimator_landing_pages_do_not_expand_sidebar_guides() -> None:
    """Keep per-estimator guide pages out of the RTD sidebar tree."""

    # core.md and other.md are section index pages; their toctrees are meant to
    # be visible in the sidebar, unlike per-estimator guide pages.
    pages = [
        page
        for page in sorted((DOCS / "estimators").glob("*.md"))
        if page.name not in {"core.md", "other.md"} and not _is_excluded_from_rtd(page)
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
        # comparison.md, landscape.md, core.md, other.md are cross-estimator
        # overview/section pages with no single source paper
        if page.name in {"comparison.md", "landscape.md", "core.md", "other.md"}:
            continue
        text = page.read_text(encoding="utf-8")
        source_pos = text.find("## Source Papers")
        if source_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Source Papers")
            continue

        first_marker_pos = min(
            (pos for marker in first_content_markers if (pos := text.find(marker)) != -1),
            default=-1,
        )
        if first_marker_pos != -1 and source_pos > first_marker_pos:
            offenders.append(f"{page.relative_to(ROOT)}: Source Papers must be near the top")

        source_section = _section_body(text, source_pos)
        if "{ref}`" not in source_section:
            offenders.append(f"{page.relative_to(ROOT)}: Source Papers must link to references")

    assert offenders == []


def test_estimator_pages_order_model_before_algorithm() -> None:
    """Lock the academic estimator-page template (2026-06-23 rebuild).

    Each public estimator main page is a self-contained academic reference: the
    model and the math live on the page, not in a separate under_the_hood
    subpage. The invariants:

    1. A ``## Model`` section containing display math (``$$``). A page with no
       model math is under-specified.
    2. A ``## Algorithm`` section with a fenced pseudocode block. A page without
       pseudocode leaves the algorithm opaque.
    3. ``## Model`` appears before ``## Algorithm``.
    """

    pages = [
        page
        for page in sorted((DOCS / "estimators").glob("*.md"))
        if page.name not in {"comparison.md", "landscape.md", "core.md", "other.md"}
        and not _is_excluded_from_rtd(page)
    ]
    assert pages

    offenders = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        model_match = re.search(r"^## Model\b", text, flags=re.MULTILINE)
        model_pos = model_match.start() if model_match else -1
        algorithm_match = re.search(r"^## Algorithm\b", text, flags=re.MULTILINE)
        algorithm_pos = algorithm_match.start() if algorithm_match else -1

        if model_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Model section")
            continue
        if "$$" not in text:
            offenders.append(f"{page.relative_to(ROOT)}: Model section has no display math")
            continue
        if algorithm_pos == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing Algorithm")
            continue
        if not model_pos < algorithm_pos:
            offenders.append(f"{page.relative_to(ROOT)}: expected Model before Algorithm")
            continue
        if text.find("```", algorithm_pos) == -1:
            offenders.append(f"{page.relative_to(ROOT)}: missing fenced code block in Algorithm")

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
        "kang-2026-lecture",
        "rawat-rust-2026",
        "kim-2021",
        "cao-2021",
    ]

    missing_ids = [
        reference_id for reference_id in expected_ids if f"({reference_id})=" not in references
    ]

    assert "   references\n" in index
    assert references.startswith("# References\n")
    assert "A Lecture Note on Offline RL and IRL" in references
    assert missing_ids == []


def test_theory_section_is_preserved_but_not_published() -> None:
    """Retain the theory sources for audit without emitting or linking them."""

    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    expected_pages = [
        "index.md",
        "soft_bellman_equivalence.md",
        "identification.md",
        "classical_ddc.md",
        "irl_boundaries.md",
        "gladius_erm.md",
        "reward_projection.md",
    ]
    pages = [DOCS / "theory" / name for name in expected_pages]

    assert all(page.exists() for page in pages)
    assert "theory/**" in _exclude_patterns()
    assert "Theory\n------" not in index
    assert "theory/index" not in index

    offenders = [
        str(page.relative_to(ROOT))
        for page in _public_doc_sources()
        if "theory/" in page.read_text(encoding="utf-8")
    ]
    assert offenders == []


def _public_doc_sources() -> list[Path]:
    roots = [
        DOCS / "index.rst",
        DOCS / "estimators.md",
        DOCS / "references.md",
        DOCS / "api" / "index.rst",
        DOCS / "estimators",
        DOCS / "theory",
        DOCS / "user_guide",
        DOCS / "simulation_studies",
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
    if path.suffix == ".md":
        opening = path.read_text(encoding="utf-8")[:80]
        if opening.startswith("---\n") and "\norphan: true\n" in opening:
            return True
    rel = path.relative_to(DOCS).as_posix()
    parts = path.relative_to(DOCS).parts
    for pattern in _exclude_patterns():
        if pattern in {"_build", "archive"} and pattern in parts:
            return True
        if fnmatch(rel, pattern):
            return True
        if pattern.endswith("/**") and (rel == pattern[:-3] or rel.startswith(pattern[:-2])):
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
