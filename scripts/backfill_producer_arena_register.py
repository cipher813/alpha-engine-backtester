#!/usr/bin/env python3
"""Derive optimizer/arena/producer_register.json from the real leaderboard history.

The selection-producer slot's arm register is a DURABLE, append-only artifact
committed to this repo — ``created_date`` drives the four-week grace period, so
a value that could silently change between cycles would make retirement
non-reproducible (champion-challenger-policy.md §6).

It is DERIVED, never typed. This script is the derivation: it reads every
``research/producer_leaderboard/{date}.json`` under the bucket, folds them in
date order through ``optimizer.producer_arena.register_events_from_boards``,
and writes the result. Re-running it after a new arm appears on the board
EXTENDS the register (earlier arms keep the ``created_date`` they were first
observed with, because the fold takes the minimum) rather than rewriting it.

    python scripts/backfill_producer_arena_register.py --bucket alpha-engine-research
    python scripts/backfill_producer_arena_register.py --from-dir ./boards --check

``--check`` writes nothing and exits non-zero if the committed artifact
disagrees with the derivation — the shape a CI guard would take.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer import producer_arena  # noqa: E402

_KEY_RE = re.compile(r"^research/producer_leaderboard/(\d{4}-\d{2}-\d{2})\.json$")


def _boards_from_s3(bucket: str) -> list[dict]:
    import boto3

    s3 = boto3.client("s3")
    boards: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="research/producer_leaderboard/"):
        for obj in page.get("Contents", []):
            if not _KEY_RE.match(obj["Key"]):
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            boards.append(json.loads(body))
    return boards


def _boards_from_dir(path: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--from-dir", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.from_dir:
        boards = _boards_from_dir(Path(args.from_dir))
    elif args.bucket:
        boards = _boards_from_s3(args.bucket)
    else:
        ap.error("one of --bucket or --from-dir is required")

    if not boards:
        # Fail loud: an empty derivation would write an empty register, which
        # resets every arm's created_date and disables the grace period pool-wide.
        print("no producer leaderboards found — refusing to write an empty register", file=sys.stderr)
        return 2

    events = producer_arena.register_events_from_boards(boards)
    payload = {
        "slot": producer_arena.SLOT,
        "derived_from": "research/producer_leaderboard/{date}.json",
        "derived_by": "scripts/backfill_producer_arena_register.py",
        "n_boards": len(boards),
        "events": events,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"

    target = producer_arena.REGISTER_PATH
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != rendered:
            print(f"{target} is STALE relative to the leaderboard history", file=sys.stderr)
            return 1
        print(f"{target} matches the derivation ({len(events)} events)")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    print(f"wrote {target} ({len(events)} events from {len(boards)} boards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
