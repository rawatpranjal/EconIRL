"""Lock the RHIP lookahead-recovery replication (Barnes et al. 2024, Figure 5).

The faithful headline: when demonstrations come from a finite-lookahead planner,
the recovery-optimal estimator horizon is interior and equals the demonstrator's
lookahead, beating both the myopic (H=0) and full-horizon (H=inf) endpoints.

This locks the property from the saved study result, so it is fast. Regenerate
the result with ``python scripts/study_rhip_lookahead.py``.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "validation" / "results" / "study_rhip_lookahead.json"
HORIZON_ORDER = ["0", "1", "2", "3", "5", "inf"]


def _mean_tv_by_demo_and_horizon() -> dict[int, dict[str, float]]:
    data = json.loads(RESULT.read_text())
    buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
    for rec in data["records"]:
        buckets[(rec["h_demo"], rec["H"])].append(rec["policy_tv"])
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for (h_demo, horizon), tvs in buckets.items():
        out[h_demo][horizon] = statistics.mean(tvs)
    return out


def test_best_fitting_horizon_recovers_demonstrator_lookahead():
    """For each demonstrator lookahead, argmin policy distance is at H = h_demo."""
    table = _mean_tv_by_demo_and_horizon()
    assert set(table) >= {1, 2, 3}, table
    for h_demo in (1, 2, 3):
        row = table[h_demo]
        best = min(row, key=row.get)
        assert best == str(h_demo), (h_demo, row)


def test_optimum_is_interior():
    """The interior optimum strictly beats both the H=0 and H=inf endpoints."""
    table = _mean_tv_by_demo_and_horizon()
    for h_demo in (1, 2, 3):
        row = table[h_demo]
        interior = row[str(h_demo)]
        assert interior < row["0"], (h_demo, "not better than myopic", row)
        assert interior < row["inf"], (h_demo, "not better than MaxEnt", row)
