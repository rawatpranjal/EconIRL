#!/usr/bin/env python3
"""Merge checkpoint shards for the official Table E.1 replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from validation.estimators.tdccp.paper_table_e1_mc import (  # noqa: E402
    _load_checkpoint,
    _summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested", type=int, default=1_000)
    args = parser.parse_args()

    records: dict[int, dict[str, object]] = {}
    for checkpoint in args.checkpoints:
        for k, record in _load_checkpoint(checkpoint).items():
            if k in records and records[k] != record:
                raise ValueError(f"conflicting checkpoint records for k={k}")
            records[k] = record

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for _, record in sorted(records.items())),
        encoding="utf-8",
    )
    payload = _summary(records, args.requested)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
