#!/usr/bin/env python3
"""Write a reproducible coverage report for every shipped localization."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
# Resolve the catalogs from this checkout even when another editable spaCR
# installation is active.  This report is a release artifact; silently reading
# a different worktree can make stale catalogs look complete.
for checkout_path in (str(ROOT), str(TOOLS)):
    while checkout_path in sys.path:
        sys.path.remove(checkout_path)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(TOOLS))
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
DISPLAY = {
    "sv": "Swedish", "de": "German", "es": "Spanish",
    "zh_CN": "Simplified Chinese", "pt": "Portuguese", "hi": "Hindi",
    "ko": "Korean", "is": "Icelandic", "fr": "French",
}


def _runtime_tables(module) -> dict[str, Mapping[str, str]]:
    return {
        "SETTING_LABELS": module.SETTING_LABELS,
        "SETTING_TOOLTIPS": module.SETTING_TOOLTIPS,
        "CATEGORY_HELP": module.CATEGORY_HELP,
        "UI": module.UI,
        "MODULE_SUMMARIES": module.MODULE_SUMMARIES,
    }


def _live_runtime_tables() -> dict[str, dict[str, str]]:
    """Extract the English runtime contract from application source."""
    from build_i18n_catalogs import canonical_sources

    sources = canonical_sources()
    return {
        "SETTING_LABELS": {
            str(key): str(value)
            for key, value in sources["setting_labels"].items()
        },
        "SETTING_TOOLTIPS": {
            str(key): str(value)
            for key, value in sources["setting_tooltips"].items()
        },
        "CATEGORY_HELP": {
            str(value): str(value) for value in sources["categories"]
        },
        "UI": {str(value): str(value) for value in sources["ui"]},
        "MODULE_SUMMARIES": {
            str(key): str(value)
            for key, value in sources["module_summaries"].items()
        },
    }


def _live_api_docstrings() -> dict[str, str]:
    """Extract the API text visible to AutoAPI from the Python source."""
    from build_documentation_i18n import public_docstrings

    return public_docstrings()


def _source_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _runtime_counts(
    source_tables: Mapping[str, Mapping[str, str]], module,
) -> tuple[int, int, int, int]:
    """Return current, total, translated and orphaned runtime counts."""
    target_tables = _runtime_tables(module)
    target_hashes = getattr(module, "SOURCE_HASHES", {})
    current = 0
    changed = 0
    orphaned = 0
    for table_name, source_table in source_tables.items():
        target_table = target_tables[table_name]
        orphaned += len(set(target_table) - set(source_table))
        for key, source in source_table.items():
            is_current = (
                key in target_table
                and target_hashes.get((table_name, key))
                == _source_hash(source)
            )
            current += is_current
            changed += is_current and target_table[key] != source
    return (
        current,
        sum(len(table) for table in source_tables.values()),
        changed,
        orphaned,
    )


def build_report() -> str:
    source_tables = _live_runtime_tables()
    api_sources = _live_api_docstrings()
    api_root = ROOT / "docs" / "source" / "_static" / "i18n" / "api"
    english_installer = json.loads((
        ROOT / "packaging" / "i18n" / "en.json"
    ).read_text(encoding="utf-8"))

    rows = []
    for language in LANGUAGES:
        module = import_module(f"spacr.qt.i18n_catalogs.{language}")
        (
            runtime_present,
            runtime_total,
            runtime_changed,
            runtime_orphaned,
        ) = _runtime_counts(source_tables, module)
        installer = json.loads((
            ROOT / "packaging" / "i18n" / f"{language}.json"
        ).read_text(encoding="utf-8"))
        api = json.loads((
            api_root / f"{language}.json"
        ).read_text(encoding="utf-8"))
        api_symbols = api.get("symbols", {})
        api_fresh = sum(
            api_symbols.get(key, {}).get("source_sha256")
            == _source_hash(source)
            and bool(api_symbols.get(key, {}).get("text", "").strip())
            for key, source in api_sources.items()
        )
        api_orphaned = len(set(api_symbols) - set(api_sources))
        readme = (
            ROOT / "docs" / "i18n" / "readme" / f"README.{language}.rst"
        )
        rows.append((
            DISPLAY[language], runtime_present, runtime_total,
            runtime_orphaned, runtime_changed, len(installer),
            len(english_installer), api_fresh, len(api_sources), api_orphaned,
            readme.stat().st_size,
        ))

    lines = [
        "# spaCR localization coverage",
        "",
        "Generated by `python tools/audit_i18n_coverage.py --write`. Exact "
        "coverage is measured against live application/API source: the key "
        "must exist and its stored source hash must be current. An unchanged "
        "count can legitimately include scientific identifiers and product "
        "names. Orphaned entries exist in a catalog but not in that live "
        "inventory.",
        "",
        "| Language | Current runtime entries | Orphaned runtime entries | Non-English current runtime values | Installer strings | Fresh API docstrings | Orphaned API docstrings | README bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (
        name, present, total, runtime_orphaned, changed, installer_count,
        installer_total, api_fresh, api_total, api_orphaned, readme_bytes,
    ) in rows:
        lines.append(
            f"| {name} | {present}/{total} | {runtime_orphaned} | "
            f"{changed}/{total} | {installer_count}/{installer_total} | "
            f"{api_fresh}/{api_total} | {api_orphaned} | {readme_bytes:,} |"
        )
    lines.extend([
        "",
        "Runtime entries comprise setting labels, authored setting tooltips, "
        "category explanations, static Qt text, and all registered module "
        "summaries. English source text remains the safe fallback.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write:
        path = ROOT / "docs" / "i18n" / "COVERAGE.md"
        path.write_text(report, encoding="utf-8")
        print(path)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
