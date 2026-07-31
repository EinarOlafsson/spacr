"""``spacr-leakage`` — audit a classifier dataset without training a model."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .classifier_evaluation import (
    audit_dataset_splits,
    write_leakage_audit,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the leakage-audit argument parser."""
    parser = argparse.ArgumentParser(
        prog="spacr-leakage",
        description=(
            "Verify that related crops do not cross a classifier train/test "
            "boundary."
        ),
    )
    parser.add_argument("dataset", help="folder containing train/ and test/")
    parser.add_argument(
        "--group-by", choices=("field", "well", "plate", "none"),
        default="well",
    )
    parser.add_argument(
        "--no-content-hash", action="store_true",
        help="skip SHA-256 detection of renamed byte-identical crops",
    )
    parser.add_argument(
        "--allow-unverifiable", action="store_true",
        help="warn instead of failing when identity or content cannot be verified",
    )
    parser.add_argument("--output", help="also write the JSON report to this path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Audit the requested dataset; return 0 pass, 1 leakage, or 2 unusable."""
    args = build_parser().parse_args(argv)
    try:
        report = audit_dataset_splits(
            args.dataset,
            group_by=args.group_by,
            hash_content=not args.no_content_hash,
            require_identity=not args.allow_unverifiable,
            raise_on_leakage=False,
        )
    except (OSError, ValueError) as exc:
        print(f"spacr-leakage: {exc}", file=sys.stderr)
        return 2
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        write_leakage_audit(args.output, report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
