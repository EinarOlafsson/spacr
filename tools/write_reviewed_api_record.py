"""Author a hand-written reviewed API translation, the way the builder reads it.

WHY THIS EXISTS. Some blocks the translation checkpoints cannot do. Not
"do badly" -- cannot: asked for Icelandic about nucleotide bases MADLAD
returned a Wikipedia stub notice, and asked for Chinese about barcode column
order it returned "we can better grasp customer needs and improve customer
satisfaction". The gates reject both, correctly, and the block then stays
exact English and the fail-closed audit stops the docs deploy.

There are two ways out and only one of them is honest. Rewriting the English
until a model can follow it degrades the page every English reader sees, and
moves the source hash so the other eight languages go stale on that symbol
too. Writing the translation by hand fixes ONE locale, leaves the English
alone, and needs no rebuild. `REVIEW_SCOPE_2026-09-04.md` records the
precedent: `spacr.ops_accel#4` in Portuguese and the `ops_gpu` tooltip in
Simplified Chinese.

The record has to be exactly right or the builder refuses it, and the two
fields that are easy to get wrong are computed here rather than typed:

  `context`        must equal `_api_translation_source(source)`, a
                   target-neutral English expansion -- NOT the raw source.
                   Wrong value raises "stale reviewed API context".
  `source_sha256`  the hash of the source block, which pins the record to
                   the English it was written against. When the English
                   later changes the record stops applying, which is the
                   correct behaviour: it is evidence about a sentence that
                   no longer exists.

A label is `<symbol>#<n>`, where n indexes `translatable_blocks(text)` -- NOT
paragraphs. Field lists split into their own blocks, so `BarcodeSpan#5` is a
`:param` line and not the sixth paragraph. This tool resolves the label the
builder's way so that distinction cannot be got wrong by counting.

    python tools/write_reviewed_api_record.py --language zh_CN \\
        --label spacr.settings.BarcodeSet#2 --translation-file zh.txt

Writes to `docs/i18n/reviewed/api/<language>/<name>.json`, merging into an
existing file when one is given. Prints the source it pinned to, so the
author can see what they translated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
CATALOGS = REPO / "docs" / "source" / "_static" / "i18n" / "api"
REVIEWED = REPO / "docs" / "i18n" / "reviewed" / "api"
REVIEWED_RUNTIME = REPO / "docs" / "i18n" / "reviewed" / "runtime"
RUNTIME_CATALOG = REPO / "spacr" / "qt" / "i18n_catalogs" / "en.py"
SCHEMA = 1


def _builder():
    """Import the builder, whose splitter and expansion define the format."""
    tools = str(REPO / "tools")
    sys.path.insert(0, tools)
    try:
        import build_documentation_i18n as module
        return module
    finally:
        sys.path.remove(tools)


def _runtime_builder():
    """Import the RUNTIME builder -- a different module from :func:`_builder`.

    The two lanes have separate builders and separate authorities for what a
    source is: `build_documentation_i18n` splits docstrings into blocks, while
    `build_i18n_catalogs.canonical_sources()` names the runtime tables. Using
    the API builder for a runtime lookup is the mistake this separate accessor
    exists to make hard.
    """
    tools = str(REPO / "tools")
    sys.path.insert(0, tools)
    try:
        import build_i18n_catalogs as module
        return module
    finally:
        sys.path.remove(tools)


def resolve(label: str) -> str:
    """The English source block a label names, split the builder's way.

    :param label: ``<symbol>#<index>``.
    :returns: the source block text.
    :raises SystemExit: when the symbol or the index does not exist, with
        the count that does, because an off-by-one here is silent.
    """
    builder = _builder()
    symbol, _, index = label.rpartition("#")
    if not symbol or not index.isdigit():
        raise SystemExit(f"label {label!r} is not <symbol>#<n>")
    english = json.loads((CATALOGS / "en.json").read_text(encoding="utf-8"))
    entry = english["symbols"].get(symbol)
    if entry is None:
        raise SystemExit(f"no symbol {symbol!r} in en.json")
    blocks, _ = builder.translatable_blocks(entry["text"])
    position = int(index)
    if position >= len(blocks):
        raise SystemExit(
            f"{symbol} has {len(blocks)} translatable blocks (0-"
            f"{len(blocks) - 1}); #{position} does not exist")
    return blocks[position]


def record(label: str, translation: str) -> dict:
    """One reviewed record, with both computed fields computed."""
    builder = _builder()
    source = resolve(label)
    return {
        "context": builder._api_translation_source(source),
        "label": label,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "translation": translation,
    }


def runtime_source(table: str, key: str) -> str:
    """The English runtime string a (table, key) names.

    RESOLVED THROUGH ``canonical_sources()``, which is the same authority
    ``reviewed_runtime_translations`` reads when it validates the record this
    writes. Resolving through the generated ``en.py`` instead does not work
    and fails confusingly: records spell the table ``ui``, the catalog module
    calls it ``UI_SOURCES``, so ``getattr(module, table.upper())`` raised "no
    table 'UI'" for the 384 existing ``ui`` records' own table name. Two
    spellings of one table is the kind of drift that makes a tool look broken
    when the evidence is fine.

    ``ui`` and ``categories`` arrive as SEQUENCES, not mappings: a static Qt
    caption has no key separate from itself, so the row IS its own English
    source and ``key`` is returned unchanged once membership is confirmed.

    :param table: the catalog table, lower case as the records spell it --
        ``ui``, ``setting_tooltips``, ``setting_labels``, ``categories``,
        ``module_summaries``, ``installer``.
    :param key: the row within it.
    :returns: the English source string.
    :raises SystemExit: when either does not exist.
    """
    sources = _runtime_builder().canonical_sources()
    table_source = sources.get(table)
    if table_source is None:
        raise SystemExit(
            f"no table {table!r}; known tables: {', '.join(sorted(sources))}"
        )
    if key not in table_source:
        raise SystemExit(f"no key {key!r} in {table}")
    if isinstance(table_source, dict):
        return str(table_source[key])
    return key


def runtime_record(table: str, key: str, translation: str) -> dict:
    """One reviewed runtime record.

    The runtime lane has NO ``context`` field -- that belongs to the API
    lane, whose sources go through an English expansion first. Adding one
    here would be rejected.
    """
    source = runtime_source(table, key)
    return {
        "key": key,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "table": table,
        "translation": translation,
    }


def write(language: str, name: str, entries: Sequence[dict]) -> Path:
    """Merge ``entries`` into a reviewed file, replacing same-label rows."""
    lane = REVIEWED_RUNTIME if "table" in entries[0] else REVIEWED
    folder = lane / language
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        document = {"language": language, "records": [], "schema": SCHEMA}
    def identity(row: dict) -> tuple:
        return (row.get("label"), row.get("table"), row.get("key"))

    replacing = {identity(entry) for entry in entries}
    kept = [r for r in document.get("records", [])
            if identity(r) not in replacing]
    document["records"] = sorted(
        kept + list(entries),
        key=lambda r: (r.get("label") or "", r.get("table") or "",
                       r.get("key") or ""))
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", required=True)
    parser.add_argument("--label", help="API lane: <symbol>#<index>")
    parser.add_argument("--table", help="runtime lane: e.g. setting_tooltips")
    parser.add_argument("--key", help="runtime lane: the row in that table")
    parser.add_argument("--translation")
    parser.add_argument("--translation-file", type=Path)
    parser.add_argument("--name", default="hand-written",
                        help="file stem under the language folder")
    parser.add_argument("--show", action="store_true",
                        help="print the source block and exit, translating "
                             "nothing -- use this first")
    args = parser.parse_args(argv)

    runtime_lane = bool(args.table or args.key)
    if runtime_lane and not (args.table and args.key):
        raise SystemExit("the runtime lane needs both --table and --key")
    if runtime_lane == bool(args.label):
        raise SystemExit("give either --label (API) or --table/--key "
                         "(runtime), not both and not neither")

    if args.show:
        print(runtime_source(args.table, args.key) if runtime_lane
              else resolve(args.label))
        return 0
    if bool(args.translation) == bool(args.translation_file):
        raise SystemExit("give exactly one of --translation, "
                         "--translation-file")
    text = (args.translation
            or args.translation_file.read_text(encoding="utf-8")).strip()
    if not text:
        raise SystemExit("the translation is empty")

    entry = (runtime_record(args.table, args.key, text) if runtime_lane
             else record(args.label, text))
    path = write(args.language, args.name, [entry])
    print(f"wrote {path}")
    print(f"  label  {entry.get('label') or entry['table'] + '/' + entry['key']}")
    print(f"  sha    {entry['source_sha256'][:16]}...")
    print(f"  source {entry['source'][:70]}...")
    checker = ("check_reviewed_runtime_evidence" if runtime_lane
               else "check_reviewed_api_evidence")
    print(f"\nVerify with: python tools/{checker}.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
