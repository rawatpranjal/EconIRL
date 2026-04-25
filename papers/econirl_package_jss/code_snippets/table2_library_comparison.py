"""Table 2: comparison of econirl with the four nearest open-source packages.
Hand-curated. The script encodes our judgments about competing libraries and
emits a LaTeX fragment.

Reproduces: figures/table2.tex.
"""
from __future__ import annotations

from pathlib import Path

ROWS = [
    ("NFXP, MPEC, CCP",                    "yes", "no",  "yes", "no",  "no"),
    ("SEES, NNES, TD-CCP",                 "yes", "no",  "no",  "no",  "no"),
    ("MCE-IRL",                            "yes", "yes", "no",  "no",  "no"),
    ("GAIL, AIRL",                         "yes", "yes", "no",  "no",  "no"),
    ("IQ-Learn, $f$-IRL",                  "yes", "no",  "no",  "no",  "no"),
    ("BC",                                 "yes", "yes", "no",  "no",  "yes"),
    ("Standard errors",                    "yes", "no",  "yes", "yes", "no"),
    ("Identification diagnostics",         "yes", "no",  "no",  "yes", "no"),
    ("Counterfactual simulation",          "yes", "no",  "no",  "yes", "no"),
    ("JAX backend",                        "yes", "no",  "no",  "no",  "no"),
]

lines = []
for r in ROWS:
    lines.append(" & ".join(r) + " \\\\")

out = Path(__file__).resolve().parents[1] / "figures" / "table2.tex"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
print("\n".join(lines))
