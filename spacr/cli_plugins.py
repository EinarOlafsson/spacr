"""Command-line diagnostics for the spaCR plugin SDK."""
from __future__ import annotations

import argparse
import json
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
        description="List and diagnose installed spaCR plugins.",
    )
    parser.add_argument(
        "command", nargs="?", choices=("list", "doctor"), default="list"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _payload() -> dict:
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
    """Run ``spacr-plugins list|doctor`` and return a shell exit code."""
    args = build_parser().parse_args(argv)
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
