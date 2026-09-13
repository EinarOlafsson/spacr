#!/usr/bin/env python3
"""Report reviewed RUNTIME records that no longer match their source string.

THE RUNTIME TWIN OF `check_reviewed_api_evidence.py`, and it exists for the
same reason. `reviewed_runtime_translations` is a hard error by design, and
since instruction 306 it collects every problem before raising rather than
stopping at the first -- but it still only runs as part of a catalog build,
and it raises rather than prints. Someone about to rename a setting wants the
list BEFORE they pay the build, and wants it as a work list rather than as a
traceback.

WHAT KNOCKS A RECORD LOOSE, and this is the part that surprises people. The
record is bound to its table, its KEY, and the exact source text with its
SHA-256. `setting_labels` and `setting_tooltips` are dicts keyed by the
SETTING KEY, so:

    RENAMING A SETTING STRANDS ITS REVIEWED RECORDS EVEN WHEN THE ENGLISH IS
    BYTE-IDENTICAL.

The loader resolves `sources[table].get(key)`, gets `None` for a key that no
longer exists, and reports `stale reviewed runtime source`. The obvious
diagnosis -- that someone edited the prose -- is wrong, and that is the most
confusing possible failure. This script tells the two apart.

WHAT THE OUTCOMES MEAN:

  MOVED    the recorded English still exists, under a different key or table.
           Nobody changed the prose; the setting was renamed. RE-BIND by
           editing the record's `key` (and `table` if it says so). The
           reviewed translation is still good and retiring it throws away
           human work for a string nobody touched.
  STALE    the recorded English is gone from every table. The setting was
           removed, or its prose was rewritten. The translation stands for
           wording the program no longer uses.
  HASH     the English matches but the recorded SHA-256 does not. The record
           was hand-edited without restamping; recompute the hash.
  GATE     source and hash agree, but the target no longer passes the ordinary
           candidate gates (non-idempotent under `_contextualize`, or rejected
           outright). These never appear as a rename's fallout; they mean the
           gates moved under a record that used to pass.
  CONFLICT two records give the same English different translations. One of
           them has to go, and the build cannot choose.

RETIRING IS A DELETION HERE, NOT AN ANNOTATION -- the one place this differs
from the API script, and getting it wrong wastes a build. The API store
retires a record in place with `"retired": "<date>"`. The runtime loader
validates `set(record) == {"table", "key", "source_sha256", "source",
"translation"}` by EXACT SET EQUALITY, so a runtime record carrying a
`retired` key is not a retired record, it is an `invalid reviewed runtime
record` and a hard failure. Remove the record from its file instead.

Exit status is 1 when anything is reported, so it can gate a push.

    python3 tools/check_reviewed_runtime_evidence.py
    python3 tools/check_reviewed_runtime_evidence.py es ko      # only these
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

REVIEWED_RUNTIME = ROOT / "docs" / "i18n" / "reviewed" / "runtime"

#: The loader's own field set, repeated here deliberately. If it ever diverges
#: this script should report a bad record rather than quietly accept one the
#: build will reject, so the duplication is the point -- and
#: `test_the_runtime_evidence_reporter_agrees_with_the_loader` pins the two
#: together.
EXPECTED_FIELDS = frozenset(
    {"table", "key", "source_sha256", "source", "translation"})


def _current_source(sources, table_name: str, key: str):
    """What the named table holds for ``key`` today, or ``None``.

    Mirrors the loader exactly: a MAPPING table is keyed by the setting key,
    and a SEQUENCE table (`ui`, `categories`) is keyed by the English string
    itself -- so for those the key and the source are the same thing.
    """
    table = sources.get(table_name)
    if isinstance(table, dict):
        return table.get(key)
    if isinstance(table, (tuple, list, set, frozenset)):
        return key if key in table else None
    return None


def _where_this_english_lives(sources) -> dict:
    """English text -> every ``(table, key)`` that carries it today.

    This is what separates MOVED from STALE. A renamed setting keeps its
    prose, so the text is still here under the new key; a removed or rewritten
    one is not here at all.
    """
    index = defaultdict(list)
    for table_name, table in sources.items():
        if isinstance(table, dict):
            for key, source in table.items():
                index[str(source)].append((table_name, str(key)))
        elif isinstance(table, (tuple, list, set, frozenset)):
            for source in table:
                index[str(source)].append((table_name, str(source)))
    return index


def _languages(argv) -> list:
    """Which locales to check: the ones named, else every one on disk."""
    if argv:
        return list(argv)
    if not REVIEWED_RUNTIME.is_dir():
        return []
    return sorted(path.name for path in REVIEWED_RUNTIME.iterdir()
                  if path.is_dir())


def main(argv=None) -> int:
    """Print every loose reviewed runtime record; return 1 if there are any."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # A FLAG IS NOT A LOCALE. `--help` was taken as a directory name and
    # reported as one malformed record, which is a confusing way to answer a
    # request for help and an actively misleading one for a typo'd locale.
    if any(a.startswith("-") for a in argv):
        if set(argv) & {"-h", "--help"}:
            print(__doc__)
            return 0
        print(f"unknown option(s): {[a for a in argv if a.startswith('-')]}\n",
              file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 2

    import build_i18n_catalogs as runtime

    sources = runtime.canonical_sources()
    lives_at = _where_this_english_lives(sources)

    moved, stale, hashes, gates, conflicts, malformed = [], [], [], [], [], []
    # source text -> (language, translation, where it was read) so a second
    # record disagreeing with the first can name both sides.
    seen_target = {}
    checked = 0

    languages = _languages(argv)
    # THE REPORTER MUST NOT RE-ENTER THE LOADER IT REPLACES, and it does
    # unless this is held. The GATE check calls `_contextualize`, which calls
    # `_reviewed_translation`, which calls `reviewed_runtime_translations` --
    # the function that RAISES on the first stale record. So a reporter run
    # against a tree that HAS stale records died with the very traceback it
    # exists to turn into a list, while a run against a clean tree looked
    # perfect. Found by simulating a rename, not by reading.
    #
    # `_REVIEWED_RUNTIME_LOADING` is the loader's OWN guard for exactly this
    # cycle -- `_reviewed_translation` returns the static answer and stops
    # when the language is in it -- so this borrows the mechanism rather than
    # inventing one. Held for the whole run and never discarded: this is a
    # one-shot script, and there is no point at which it wants the loader to
    # start raising again.
    runtime._REVIEWED_RUNTIME_LOADING.update(languages)

    for language in languages:
        directory = REVIEWED_RUNTIME / language
        if not directory.is_dir():
            malformed.append(f"  {language}  (no such reviewed directory)")
            continue
        for path in sorted(directory.glob("*.json")):
            where = f"{language}/{path.name}"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                malformed.append(f"  {where}  (unreadable: {exc})")
                continue
            if payload.get("schema") != 1 or payload.get("language") != language:
                malformed.append(f"  {where}  (bad header)")
                continue
            records = payload.get("records")
            if not isinstance(records, list):
                malformed.append(f"  {where}  (records is not a list)")
                continue
            for record in records:
                if not isinstance(record, dict) or set(record) != EXPECTED_FIELDS:
                    extra = (sorted(set(record) - EXPECTED_FIELDS)
                             if isinstance(record, dict) else [])
                    note = f"unexpected field(s) {extra}" if extra else "bad fields"
                    # `retired` is the one people reach for by analogy with
                    # the API store, where it is correct. Here it is a build
                    # failure, so say what to do instead.
                    if "retired" in extra:
                        note = ("carries 'retired' -- the runtime loader "
                                "validates the field set exactly, so this is "
                                "an invalid record, not a retired one. "
                                "DELETE the record instead")
                    malformed.append(f"  {where}  ({note})")
                    continue
                checked += 1
                table_name = str(record["table"])
                key = str(record["key"])
                source = str(record["source"])
                target = str(record["translation"])
                current = _current_source(sources, table_name, key)
                if current != source:
                    homes = [(t, k) for t, k in lives_at.get(source, ())
                             if (t, k) != (table_name, key)]
                    if homes:
                        new_table, new_key = homes[0]
                        rebind = (f"key -> {new_key!r}" if new_table == table_name
                                  else f"table/key -> {new_table}/{new_key!r}")
                        moved.append(
                            f"  {where}  {table_name}/{key}  -> re-bind {rebind}")
                    else:
                        why = ("key is gone" if current is None
                               else "prose was rewritten")
                        stale.append(f"  {where}  {table_name}/{key}  ({why})")
                    continue
                if record["source_sha256"] != hashlib.sha256(
                        source.encode("utf-8")).hexdigest():
                    hashes.append(f"  {where}  {table_name}/{key}")
                    continue
                if runtime._contextualize(target, language, source) != target:
                    gates.append(
                        f"  {where}  {table_name}/{key}  (non-idempotent)")
                    continue
                reasons = runtime._translation_rejection_reasons(
                    source, target, language,
                    force=runtime._looks_translatable(source))
                if reasons:
                    first = reasons[0] if isinstance(reasons, (list, tuple)) else reasons
                    gates.append(f"  {where}  {table_name}/{key}  ({first})")
                    continue
                previous = seen_target.setdefault(
                    (language, source), (target, where))
                if previous[0] != target:
                    conflicts.append(
                        f"  {where}  {table_name}/{key}  disagrees with "
                        f"{previous[1]} on {source[:48]!r}")

    reported = 0
    for title, rows in (
        ("MOVED -- re-bind the key, do NOT delete", moved),
        ("STALE -- the English is gone; DELETE the record (no 'retired' field "
         "here)", stale),
        ("HASH -- English matches, recorded SHA-256 does not; restamp", hashes),
        ("GATE -- source is fine, the target no longer passes the gates", gates),
        ("CONFLICT -- one English string, two translations", conflicts),
        ("MALFORMED -- the loader will refuse these outright", malformed),
    ):
        if rows:
            reported += len(rows)
            print(f"{title} ({len(rows)}):")
            print("\n".join(rows))
            print()
    # CHECKING NOTHING IS NOT PASSING, and this is the failure shape that
    # survives longest: a guard that silently reports zero problems because it
    # found zero records looks identical to a clean tree. It used to print
    # "1 of 0 records need attention", which is not a sentence about anything.
    if checked == 0:
        print(f"NOTHING WAS CHECKED. {len(languages)} locale(s) requested "
              f"({', '.join(languages) or 'none'}) and no reviewed record was "
              f"read -- so this run says nothing about whether the evidence is "
              f"stale. Check the locale names and that "
              f"{REVIEWED_RUNTIME} is populated.", file=sys.stderr)
        if malformed:
            print("\n".join(malformed), file=sys.stderr)
        return 2

    if not reported:
        print(f"every reviewed runtime record still matches its source "
              f"({checked} checked)")
        return 0
    print(f"{reported} of {checked} reviewed runtime records need attention")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
