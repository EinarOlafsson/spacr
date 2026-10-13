"""Register translation debt without making localized coverage a release gate.

English source correctness remains a separate required check. This report is
safe to run against either branch: builders are loaded from that checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
from string import Formatter


def api_issues(english: dict, localized: dict) -> list[dict]:
    """Return every incompatible symbol, including missing and obsolete ones."""
    issues = []
    if localized.get("schema") != 2:
        issues.append({"kind": "schema", "expected": 2})
    expected = english["symbols"]
    actual = localized.get("symbols", {})
    if not isinstance(actual, dict):
        return issues + [{"kind": "malformed_symbols"}]
    for key in sorted(set(expected) | set(actual)):
        if key not in actual:
            issues.append({"kind": "missing", "symbol": key})
        elif key not in expected:
            issues.append({"kind": "obsolete", "symbol": key})
        else:
            record = actual[key]
            if not isinstance(record, dict) or not str(record.get("text", "")).strip():
                issues.append({"kind": "empty_or_malformed", "symbol": key})
                continue
            for field in ("source_sha256", "source_blocks_sha256"):
                if record.get(field) != expected[key].get(field):
                    issues.append({"kind": "stale", "symbol": key, "field": field})
            hashes = record.get("translation_source_blocks_sha256")
            if (not isinstance(hashes, list) or
                    len(hashes) != len(expected[key].get("source_blocks_sha256", []))):
                issues.append({"kind": "translation_blocks", "symbol": key})
    return issues


def audit_record(builder, sources, languages) -> dict:
    """Keep auditor failures and crashes visible without confusing them with passes."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = builder.audit(sources, languages)
    except Exception as error:
        return {"status": "audit_error", "error": f"{type(error).__name__}: {error}",
                "diagnostics": output.getvalue()}
    return {"status": "compatible" if status == 0 else "incompatible",
            "exit_code": status, "diagnostics": output.getvalue()}


def runtime_issues(sources: dict, localized: dict, source_hashes: dict) -> list[dict]:
    """Register missing/stale rows and incompatible format placeholders."""
    issues = []
    formatter = Formatter()
    for table, entries in sources.items():
        actual = localized.get(table, {})
        if not isinstance(actual, dict):
            issues.append({"kind": "malformed_table", "table": table})
            continue
        for key in sorted(set(entries) | set(actual)):
            issue = {"table": table, "key": key}
            if key not in actual:
                issues.append({**issue, "kind": "missing"})
            elif key not in entries:
                issues.append({**issue, "kind": "obsolete"})
            else:
                value = actual[key]
                if not isinstance(value, str) or not value.strip():
                    issues.append({**issue, "kind": "empty_or_malformed"})
                    continue
                hash_key = (table, str(key))
                if localized.get("SOURCE_HASHES", {}).get(hash_key) != source_hashes.get(hash_key):
                    issues.append({**issue, "kind": "stale"})
                try:
                    fields = lambda text: sorted(field for _, field, _, _ in formatter.parse(text)
                                                 if field is not None)
                    if fields(entries[key]) != fields(value):
                        issues.append({**issue, "kind": "placeholders"})
                except ValueError:
                    issues.append({**issue, "kind": "malformed_placeholders"})
    return issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--english-required", action="store_true")
    parser.add_argument("--full-audit", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tools"))
    report = {"schema": 1, "source_commit": os.environ.get("GITHUB_SHA", ""),
              "policy": "translation incompatibilities are report-only",
              "scope": "source contracts and runtime placeholders",
              "full_audit": args.full_audit,
              "english": {}, "api": {}, "runtime": {}}
    docs = importlib.import_module("build_documentation_i18n")
    report["retired_readme_reviews"] = [
        {"source": source, "languages": sorted(translations),
         "reason": "English paragraph no longer appears in the current README"}
        for source, translations in sorted(
            getattr(docs, "RETIRED_README_EVIDENCE_BLOCKS", {}).items())
    ]
    runtime = importlib.import_module("build_i18n_catalogs")
    for name, builder, extractor in (("api", docs, docs.public_docstrings),
                                      ("runtime", runtime, runtime.canonical_sources)):
        sources = extractor()
        report["english"][name] = audit_record(builder, sources, ())
        if args.full_audit:
            for language in builder.MODEL_SPECS:
                report[name][language] = audit_record(builder, sources, (language,))
    api_dir = root / "docs/source/_static/i18n/api"
    english = json.loads((api_dir / "en.json").read_text())
    for language in docs.MODEL_SPECS:
        row = report["api"].setdefault(language, {})
        try:
            payload = json.loads((api_dir / f"{language}.json").read_text())
            row["issues"] = api_issues(english, payload)
            row["source_compatible"] = not row["issues"]
        except (OSError, ValueError, AttributeError) as error:
            row.update(source_compatible=False, issues=[{
                "kind": "unreadable_catalog", "error": str(error)}])
    sources = runtime.canonical_sources()
    tables = {"SETTING_LABELS": sources["setting_labels"],
              "SETTING_TOOLTIPS": sources["setting_tooltips"],
              "CATEGORY_HELP": {key: key for key in sources["categories"]},
              "UI": {key: key for key in sources["ui"]},
              "MODULE_SUMMARIES": sources["module_summaries"]}
    for language in runtime.MODEL_SPECS:
        row = report["runtime"].setdefault(language, {})
        try:
            path = root / "spacr/qt/i18n_catalogs" / f"{language}.py"
            namespace = {}
            exec(compile(path.read_text(), str(path), "exec"), namespace)
            row["issues"] = runtime_issues(tables, namespace, runtime._source_hashes(sources))
            row["source_compatible"] = not row["issues"]
        except Exception as error:
            row.update(source_compatible=False, issues=[{
                "kind": "unreadable_catalog", "error": str(error)}])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    incompatible = sum(not row["source_compatible"] for row in report["api"].values())
    print(f"Registered {incompatible} incompatible API locales in {args.output}; report-only.")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(f"\nTranslation compatibility: {incompatible} API locales use English fallback. "
                         "Full diagnostics are in the translation report artifact.\n")
    if args.english_required and any(
        row["status"] != "compatible" for row in report["english"].values()
    ):
        print("English source audit failed; see the report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
