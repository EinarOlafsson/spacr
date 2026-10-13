"""Command-line diagnostics and catalogue installs for the spaCR plugin SDK.

``spacr-plugins list`` and ``doctor`` report what is installed and what failed
to load. ``catalogue`` browses a plugin and recipe catalogue (a JSON file,
its folder or an address, or ``SPACR_PLUGIN_CATALOGUE``), and ``install``,
``update`` and ``uninstall`` manage one entry of it by key.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .plugins import (
    PLUGIN_API_VERSION,
    diagnostics,
    discover_plugins,
    plugin_apps,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``spacr-plugins`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="spacr-plugins",
        description="List and diagnose installed spaCR plugins, and browse, "
                    "install, update or uninstall catalogue plugins and "
                    "recipes.",
    )
    parser.add_argument(
        "command", nargs="?", default="list",
        choices=("list", "doctor", "catalogue", "install", "update",
                 "uninstall"),
    )
    parser.add_argument("key", nargs="?", default="",
                        help="catalogue entry to install, update or uninstall")
    parser.add_argument("--catalogue", default=None,
                        help="catalogue file, folder or address")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _catalogue_command(args: argparse.Namespace) -> int:
    """Run a catalogue subcommand and return its exit code."""
    from . import plugins

    try:
        if args.command == "catalogue":
            rows = plugins._catalogue_rows(args.catalogue)
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
                return 0
            for row in rows:
                have = f", installed {row['installed']}" if row["installed"] else ""
                print(f"[{row['kind']}] {row['key']}: {row['name']} "
                      f"{row['version']} ({row['status']}{have})")
                print(f"  by {row['author'] or 'unknown'}; licence "
                      f"{row['licence'] or 'not stated'}")
                if row["summary"]:
                    print(f"  {row['summary']}")
            return 0
        if not args.key:
            raise ValueError(f"{args.command} needs the key of a catalogue entry")
        if args.command == "uninstall":
            if plugins._uninstall_from_catalogue(args.key):
                print(f"Uninstalled {args.key}.")
                return 0
            print(f"{args.key} is not installed from a catalogue.", file=sys.stderr)
            return 1
        record = plugins._install_from_catalogue(args.key, args.catalogue)
        print(f"Installed {record['name']} {record['version']} to {record['path']}")
        return 0
    except Exception as exc:
        print(f"spacr-plugins {args.command}: {exc}", file=sys.stderr)
        return 1


def _payload() -> dict:
    """Collect installed plugin contributions and discovery diagnostics.

    :returns: JSON-serializable plugin SDK version, plugins, apps, and
        diagnostics for console or machine-readable output.
    """
    installed = discover_plugins()
    return {
        "sdk_version": PLUGIN_API_VERSION,
        "plugins": [
            {
                "name": plugin.name,
                "version": plugin.version,
                "api_version": plugin.api_version,
                "apps": [app.key for app in plugin.apps],
                "model_providers": [
                    provider.key for provider in plugin.model_providers
                ],
                "report_sections": [
                    section.key for section in plugin.report_sections
                ],
            }
            for plugin in installed
        ],
        "apps": [app.key for app in plugin_apps()],
        "diagnostics": [
            {
                "plugin": item.plugin,
                "severity": item.severity,
                "message": item.message,
                "exception": item.exception,
            }
            for item in diagnostics()
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``spacr-plugins`` and return a shell exit code.

    :param argv: arguments without the program name; ``sys.argv`` when None.
    """
    args = build_parser().parse_args(argv)
    if args.command not in ("list", "doctor"):
        return _catalogue_command(args)
    payload = _payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"spaCR plugin SDK {payload['sdk_version']}")
        if not payload["plugins"]:
            print("No plugins discovered.")
        for plugin in payload["plugins"]:
            print(f"{plugin['name']} {plugin['version']} (API {plugin['api_version']})")
            for field in ("apps", "model_providers", "report_sections"):
                values = ", ".join(plugin[field]) or "none"
                print(f"  {field.replace('_', ' ')}: {values}")
        if payload["diagnostics"]:
            print("Diagnostics:")
            for item in payload["diagnostics"]:
                suffix = f" — {item['exception']}" if item["exception"] else ""
                print(
                    f"  [{item['severity']}] {item['plugin']}: "
                    f"{item['message']}{suffix}"
                )
        elif args.command == "doctor":
            print("No plugin errors recorded.")
    return 1 if args.command == "doctor" and payload["diagnostics"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
