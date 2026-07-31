"""Command-line SQLite health and concurrency audit."""
from __future__ import annotations

import argparse
import json
import sys

from .database_concurrency import inspect_database, run_concurrency_probe


def build_parser() -> argparse.ArgumentParser:
    """Build the ``spacr-db-audit`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="spacr-db-audit",
        description=(
            "Inspect a spaCR SQLite database and optionally run a destructive "
            "reader/writer stress test on a separate new scratch database."),
    )
    parser.add_argument(
        "database", nargs="?",
        help="existing database to inspect (never modified)")
    parser.add_argument(
        "--quick-check", action="store_true",
        help="run SQLite PRAGMA quick_check on the inspected database")
    parser.add_argument(
        "--probe", action="store_true",
        help="run a concurrent reader/writer probe in a disposable database")
    parser.add_argument(
        "--scratch",
        help="new, non-existing probe database path; default is temporary")
    parser.add_argument("--writers", type=int, default=4)
    parser.add_argument("--readers", type=int, default=3)
    parser.add_argument("--writes", type=int, default=50,
                        help="commits per writer")
    parser.add_argument(
        "--journal-mode", choices=("WAL", "DELETE"), default="WAL")
    parser.add_argument("--json", action="store_true",
                        help="emit one machine-readable JSON document")
    return parser


def main(argv=None) -> int:
    """Inspect/probe SQLite and return zero only when every requested check passes."""
    args = build_parser().parse_args(argv)
    if not args.database and not args.probe:
        build_parser().error("provide DATABASE, --probe, or both")
    payload = {}
    ok = True
    try:
        if args.database:
            health = inspect_database(
                args.database, quick_check=args.quick_check)
            payload["database"] = health.to_dict()
            if health.quick_check not in (None, "ok"):
                ok = False
        if args.probe:
            probe = run_concurrency_probe(
                args.scratch,
                writers=args.writers,
                readers=args.readers,
                writes_per_writer=args.writes,
                journal_mode=args.journal_mode,
            )
            payload["probe"] = probe.to_dict()
            ok = ok and probe.ok
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        ok = False

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if "database" in payload:
            item = payload["database"]
            print(
                f"Database: {item['path']}\n"
                f"  journal={item['journal_mode']} "
                f"filesystem={item['filesystem'] or 'unknown'} "
                f"busy_timeout={item['busy_timeout_ms']} ms")
            if item["quick_check"] is not None:
                print(f"  quick_check={item['quick_check']}")
            for warning in item["warnings"]:
                print(f"  WARNING: {warning}")
        if "probe" in payload:
            item = payload["probe"]
            print(
                f"Probe: {'PASS' if item['ok'] else 'FAIL'} "
                f"{item['actual_rows']}/{item['expected_rows']} rows, "
                f"{item['reader_queries']} concurrent reads, "
                f"{item['duration_seconds']:.3f} s, "
                f"journal={item['journal_mode']}")
            for error in item["errors"]:
                print(f"  ERROR: {error}")
        if "error" in payload:
            print(f"ERROR: {payload['error']}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
