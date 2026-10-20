"""Preserve a retained coverage measurement and disclose its original gate state."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def verify_report(original: Path, regenerated: Path | None = None) -> str:
    """Return the original coverage.py version after checking replay integrity."""
    report = json.loads((original / "coverage.json").read_text(encoding="utf-8"))
    gate = json.loads(
        (original / "module-coverage-ratchet.json").read_text(encoding="utf-8")
    )
    integrity = gate.get("measurement_integrity", {})
    if (integrity.get("checked") is not True
            or integrity.get("status") not in {"complete", "incomplete"}
            or not isinstance(integrity.get("issues"), list)
            or integrity.get("shard_count") != 12
            or integrity.get("shards_with_records") != 12):
        raise ValueError("The original twelve-shard measurement integrity record is missing")
    version = report.get("meta", {}).get("version", "")
    if not isinstance(version, str) or not re.fullmatch(r"7\.[0-9]+\.[0-9]+", version):
        raise ValueError("The original coverage.py version is missing or unsupported")
    if report["meta"].get("branch_coverage") is not True or not report.get("files"):
        raise ValueError("The original report must contain branch coverage")
    if regenerated is not None:
        replay = json.loads(regenerated.read_text(encoding="utf-8"))
        if (replay.get("totals") != report.get("totals")
                or replay.get("files") != report["files"]
                or replay.get("meta", {}).get("version") != version
                or replay.get("meta", {}).get("branch_coverage") is not True):
            raise ValueError("Regenerated coverage differs from the original measurement")
    print(f"Original combined coverage: {report['totals']['percent_covered']:.8f}%")
    print(f"Original measurement integrity: {integrity['status']}")
    print(f"Original coverage gate: {gate.get('status', 'unknown')}")
    for issue in integrity["issues"]:
        print(f"Original measurement issue: {issue}")
    print("Informational replay preserves the original measurement; it does not make CI or the gate pass.")
    return version


def main() -> int:
    """Validate a downloaded report before installation and again before upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--regenerated", type=Path)
    args = parser.parse_args()
    version = verify_report(args.original, args.regenerated)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"coverage_version={version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
