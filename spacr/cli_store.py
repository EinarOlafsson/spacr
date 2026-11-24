"""``spacr store``: check, fold, compare, export and migrate a project (item 0aj, S8).

``spacr store compact|verify|compare|export|ui|migrate`` are thin wrappers
over :mod:`spacr.store` and :mod:`spacr.image_store`; none holds logic of its
own, so the command line and the GUI run the same code.

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
    verify.add_argument("--full", action="store_true",
                        help=tr("rehash every catalogued table file, not only "
                                "its size and time"))
    compact = commands.add_parser(
        "compact", help=tr("fold the edit log of a store project into snapshots"))
    compact.add_argument("project", help=tr("the store project folder"))
    compact.add_argument("--json", action="store_true",
                         help=tr("print one machine-readable JSON document"))
    compare = commands.add_parser(
        "compare", help=tr("compare the tables of two store projects"))
    compare.add_argument("first", help=tr("the first project"))
    compare.add_argument("second", help=tr("the second project"))
    compare.add_argument("--table", action="append",
                         help=tr("compare only this table (repeatable)"))
    compare.add_argument("--json", action="store_true",
                         help=tr("print one machine-readable JSON document"))
    export = commands.add_parser(
        "export", help=tr("write tables to SQLite, CSV or Parquet"))
    export.add_argument("project", help=tr("the store project folder"))
    export.add_argument("target", help=tr("the file to write"))
    export.add_argument("--format", choices=("sqlite", "csv", "parquet"),
                        help=tr("file format; read from the target's suffix when omitted"))
    export.add_argument("--table", action="append",
                        help=tr("table to export (repeatable; CSV and Parquet take one)"))
    export.add_argument("--json", action="store_true",
                        help=tr("print one machine-readable JSON document"))
    ui = commands.add_parser(
        "ui", help=tr("open the project's DuckDB catalog in the duckdb shell"))
    ui.add_argument("project", help=tr("the store project folder"))
    ui.add_argument("--print", dest="show", action="store_true",
                    help=tr("only print the duckdb command"))
    migrate = commands.add_parser(
        "migrate", help=tr("convert a classic measurements.db into a store project"))
    migrate.add_argument("db", help=tr("the measurements.db, or the folder holding it"))
    migrate.add_argument("--json", action="store_true",
                         help=tr("print one machine-readable JSON document"))
    return parser


def _emit(args, payload, lines) -> int:
    """Print ``payload`` as JSON or ``lines`` as text; exit 0."""
    if args.json:
        print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    else:
        for line in lines:
            print(line)
    return 0


def _fail(args, label, exc) -> int:
    """Report a failed command; exit 1."""
    from .qt.i18n import tr

    if getattr(args, "json", False):
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    else:
        print(tr("store {label} FAILED: {error}", label=label, error=exc),
              file=sys.stderr)
    return 1


def _run_other(args) -> int:
    """Run the subcommands other than ``verify``."""
    import os
    import shutil

    from . import store
    from .qt.i18n import tr

    command = args.command
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if command == "compact":
                result = store.compact(args.project)
            elif command == "compare":
                result = store.compare(args.first, args.second, tables=args.table)
            elif command == "export":
                fmt = args.format or os.path.splitext(args.target)[1].lstrip(".").lower()
                result = store.export(args.project, args.target, fmt=fmt,
                                      tables=(args.table[0] if args.table and len(args.table) == 1
                                              and fmt != "sqlite" else args.table))
            elif command == "migrate":
                result = store.migrate_sqlite(args.db)
            else:
                path = os.path.join(store.locate(args.project), store.CATALOG)
                if not os.path.isfile(path):
                    raise store.StoreError(tr("{path} holds no {name}",
                                              path=args.project, name=store.CATALOG))
                result = {"command": ["duckdb", "-readonly", path]}
    except Exception as exc:
        return _fail(args, command, exc)
    if command == "compact":
        return _emit(args, result, [tr(
            "folded {journals} journal(s); {files} edit file(s) became snapshots of {rows} row(s)",
            journals=result["journals"], files=result["files"], rows=result["rows"])])
    if command == "export":
        return _emit(args, result, [tr("wrote {rows} row(s) to {path}",
                                       rows=result["rows"], path=result["path"])]
                     + [f"  {t}: {n}" for t, n in result["tables"].items()])
    if command == "migrate":
        lines = [tr("store project: {path}", path=result["project"])]
        lines += [f"  {t}: {c['source']} -> {c['store']} rows"
                  for t, c in result["tables"].items()]
        return _emit(args, result, lines + [tr("OK: every row count matches.")])
    if command == "ui":
        text = " ".join(result["command"])
        if args.show or shutil.which("duckdb") is None:
            print(text)
            return 0
        import subprocess

        return subprocess.call(result["command"])
    return _compare_report(args, result)


def _compare_report(args, result) -> int:
    """Print :func:`spacr.store.compare`'s report; exit 0 only when equal."""
    from .qt.i18n import tr

    lines = [tr("only in the first project: {names}", names=", ".join(result["only_first"]))
             ] if result["only_first"] else []
    if result["only_second"]:
        lines.append(tr("only in the second project: {names}",
                        names=", ".join(result["only_second"])))
    for name, entry in result["tables"].items():
        lines.append(f"{name}: rows {entry['rows'][0]} / {entry['rows'][1]}"
                     f" {'OK' if entry['ok'] else 'DIFFERS'}")
        for column in entry["only_first"] + entry["only_second"]:
            lines.append(f"  column in one project only: {column}")
        for column, found in entry["columns"].items():
            if not found["ok"]:
                lines.append(f"  {column}: max diff {found['max_diff']}, tolerance "
                             f"{found['tolerance']:.3g}, {found['differing']} differing")
    lines.append(tr("OK: the projects agree.") if result["ok"]
                 else tr("FAILED: the projects differ."))
    _emit(args, dict(result), lines)
    return 0 if result["ok"] else 1


def _verify_catalog(args):
    """Check the table catalog of a store project; None when it holds none.

    A project with a catalog and no ``fov.ome.zarr`` is checked here alone;
    one with both is checked here first and its images after, by the caller.

    :returns: None to carry on with the image check, else the exit code.
    """
    import os

    from . import store
    from .qt.i18n import tr

    path = os.path.abspath(args.project)
    if not os.path.isfile(os.path.join(path, store.CATALOG)):
        return None
    try:
        held = store.open(path, verify="none")
        try:
            files = held.verify(full=args.full)
        finally:
            held.close()
    except Exception as exc:
        return _fail(args, "verify", exc)
    images = os.path.isdir(os.path.join(path, "fov.ome.zarr"))
    if args.json and not images:
        print(json.dumps({"ok": True, "catalog_files": files}))
    elif not args.json:
        print(tr("catalog files checked: {n}", n=files))
        if not images:
            print(tr("OK: every catalogued file matches."))
    return None if images else 0


def main(argv=None) -> int:
    """Run ``spacr store``.

    :param argv: the arguments after ``store``; None reads the process arguments.
    :returns: 0 when every plane matches, 1 on any mismatch or error.
    :raises SystemExit: status 2 for invalid arguments.
    """
    from .image_store import verify_project
    from .qt.i18n import tr

    args = build_parser().parse_args(argv)
    if args.command != "verify":
        return _run_other(args)
    catalog = _verify_catalog(args)
    if catalog is not None:
        return catalog
    try:
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
