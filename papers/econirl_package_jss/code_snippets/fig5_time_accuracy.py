"""Figure 5: time-vs-accuracy frontier. Reads the cross-estimator benchmark
CSV and plots cosine similarity to ground truth against log wall-clock time.

Reproduces: figures/fig5_time_accuracy.pdf.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "examples" / "rust-bus-engine" / "benchmark_results.csv"

names, times, sims = [], [], []
with open(CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            t = float(row["time_seconds"])
            s = float(row["cosine_similarity"])
        except (KeyError, ValueError):
            continue
        names.append(row["estimator"])
        times.append(t)
        sims.append(s)

times = np.asarray(times)
sims = np.asarray(sims)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(times, sims, color="#1F4E79", s=40)
for n, t, s in zip(names, times, sims):
    ax.annotate(n, (t, s), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Wall-clock time (seconds, log scale)")
ax.set_ylabel("Cosine similarity to ground truth")
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "fig5_time_accuracy.pdf"
fig.savefig(out)
print(f"wrote {out}")
