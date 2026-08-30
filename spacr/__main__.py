"""Command-line entry point for ``python -m spacr``.

The module builds and dispatches spaCR's command-line subcommands. With no
subcommand it opens the legacy Tk interface; use ``python -m spacr.qt`` or the
installed ``spacr`` command to open the PySide6 application.

Copyright © 2025 olafsson lab
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser exposing the ``spacr`` CLI subcommands.

    :returns: Parser accepting a single positional ``command`` argument.
    """
    parser = argparse.ArgumentParser(prog="spacr")
    parser.add_argument(
        "command",
        nargs="?",
        default="gui",
        choices=[
            "gui",
            "mask",
            "measure",
            "classify",
            "annotate",
            "sequencing",
            "umap",
            "make-masks",
            "version",
        ],
        help="Command to run.",
    )
    return parser


#: The subcommand a user types -> the Qt tab it should open on.
#:
#: `gui` names no tab: it opens on Home, which is what it always did.
_APP_KEYS = {
    "mask": "mask",
    "measure": "measure",
    "classify": "classify_merged",
    "annotate": "annotate",
    "sequencing": "map_barcodes",
    "umap": "umap",
    "make-masks": "make_masks",
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; dispatch to the requested spacr subcommand.

    :param argv: Argument list to parse. When None, ``sys.argv[1:]`` is used.
    :returns: Process exit code, 0 on success.
    :raises SystemExit: with code 2 for a command the parser accepts and the
        dispatch below does not know -- `parser.error` never returns, so
        there is no exit code to hand back for that case.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        from .version import version_str
        print(version_str)
        return 0

    # EVERY WINDOW COMMAND OPENS THE Qt APPLICATION. The seven Tk screens
    # these used to start are tabs in it, so a script that still says
    # `python -m spacr mask` lands on the Mask tab rather than failing to
    # import a module that no longer exists.
    if args.command in ("gui", "mask", "measure", "classify", "annotate",
                        "sequencing", "umap", "make-masks"):
        from .qt import run

        # `run` takes the argv the launcher would have had, and its first
        # positional IS the screen to open on.
        key = _APP_KEYS.get(args.command)
        return int(run([key] if key else []) or 0)

    # `parser.error` is annotated NoReturn and raises SystemExit(2); a
    # `return 2` after it is unreachable, and an unreachable line is a line
    # no test can ever justify.
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
