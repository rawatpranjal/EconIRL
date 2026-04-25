"""Table 1: estimator taxonomy. Reads econirl.estimation.categories and emits
the LaTeX rows for the taxonomy table in Section 2.

Reproduces: figures/table1.tex.
"""
from __future__ import annotations

from pathlib import Path

from econirl.estimation.categories import ESTIMATOR_REGISTRY

ROW_FMT = "{name} & {family} & {reward} & {trans} & {inner} & {cont} & {recovers} \\\\"


def fmt_yn(b):
    return "yes" if b else "no"


lines = []
for name, info in ESTIMATOR_REGISTRY.items():
    caps = info.capabilities if hasattr(info, "capabilities") else info
    lines.append(ROW_FMT.format(
        name=name,
        family=getattr(caps, "family", ""),
        reward=getattr(caps, "reward_form", ""),
        trans=fmt_yn(getattr(caps, "requires_transitions", False)),
        inner=fmt_yn(getattr(caps, "requires_inner_solve", False)),
        cont=fmt_yn(getattr(caps, "supports_continuous_states", False)),
        recovers=fmt_yn(getattr(caps, "recovers_parameters", False)),
    ))

out = Path(__file__).resolve().parents[1] / "figures" / "table1.tex"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
print("\n".join(lines))
