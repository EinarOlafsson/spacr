"""``spacr store``: check a project's image store from the command line (item 0aj).

``spacr store verify <project>`` (also ``spacr-store verify <project>``)
re-hashes every channel plane of ``<project>/fov.ome.zarr`` against the
sha256 recorded when it was written and, when the original images are still
in the project folder or ``orig/``, compares each stored plane with its
original, value, dtype and shape. It exits 0 only when every plane matches.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the ``spacr store`` argument parser.

    :returns: the parser, with the ``verify`` subcommand.
    """
    from .qt.i18n import tr

    parser = argparse.ArgumentParser(
        prog="spacr store",
        description=tr("Check a spaCR project's fov.ome.zarr image store."))
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify", help=tr("check every stored channel plane against its recorded "
                          "checksum and, when present, its original image"))
    verify.add_argument("project", help=tr("the project folder or its fov.ome.zarr"))
    verify.add_argument("--no-originals", action="store_true",
                        help=tr("check the recorded checksums only"))
    verify.add_argument("--json", action="store_true",
                        help=tr("print one machine-readable JSON document"))
    return parser


def main(argv=None) -> int:
    """Run ``spacr store``.

    :param argv: the arguments after ``store``; None reads the process arguments.
    :returns: 0 when every plane matches, 1 on any mismatch or error.
    :raises SystemExit: status 2 for invalid arguments.
    """
    from .image_store import verify_project
    from .qt.i18n import tr

    args = build_parser().parse_args(argv)
    try:
        # The library's progress lines go to stderr, so stdout is the report.
        with contextlib.redirect_stdout(sys.stderr):
            report = verify_project(args.project, originals=not args.no_originals)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        else:
            print(tr("store verify FAILED: {error}", error=exc), file=sys.stderr)
        return 1
    ok = not report["mismatches"]
    if args.json:
        print(json.dumps(dict(report, ok=ok), indent=1, sort_keys=True))
        return 0 if ok else 1
    print(tr("fields checked: {n}", n=report["fields"]))
    print(tr("planes checked against their recorded checksums: {n}", n=report["planes"]))
    print(tr("planes compared with their original images: {n}",
             n=report["originals_compared"]))
    for name in report["unrecorded"]:
        print(tr("no checksum recorded (written before checksums): {name}", name=name))
    for note in report["notes"]:
        print(tr("note: {note}", note=note))
    for problem in report["mismatches"]:
        print(tr("MISMATCH: {problem}", problem=problem))
    if ok:
        print(tr("OK: every stored channel plane matches."))
        return 0
    print(tr("FAILED: {n} mismatched plane(s).", n=len(report["mismatches"])))
    return 1


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    sys.exit(main())
