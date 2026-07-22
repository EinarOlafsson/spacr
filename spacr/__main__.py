"""
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; dispatch to the requested spacr subcommand.

    :param argv: Argument list to parse. When None, ``sys.argv[1:]`` is used.
    :returns: Process exit code (0 on success, 2 on unknown command).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        from .version import version_str
        print(version_str)
        return 0

    if args.command == "gui":
        from .gui import gui_app
        gui_app()
        return 0

    if args.command == "mask":
        from .app_mask import start_mask_app
        start_mask_app()
        return 0

    if args.command == "measure":
        from .app_measure import start_measure_app
        start_measure_app()
        return 0

    if args.command == "classify":
        from .app_classify import start_classify_app
        start_classify_app()
        return 0

    if args.command == "annotate":
        from .app_annotate import start_annotate_app
        start_annotate_app()
        return 0

    if args.command == "sequencing":
        from .app_sequencing import start_seq_app
        start_seq_app()
        return 0

    if args.command == "umap":
        from .app_umap import start_umap_app
        start_umap_app()
        return 0

    if args.command == "make-masks":
        from .app_make_masks import start_make_mask_app
        start_make_mask_app()
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())