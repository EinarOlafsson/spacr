"""Verify only N655's recorded October 12 caption/API acceptance debt."""
from __future__ import annotations

import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
api = importlib.import_module("build_documentation_i18n")
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(relative):
    return json.loads((ROOT / relative).read_text())


def documented_nodes(tree, prefix):
    result = {}
    for node in ast.iter_child_nodes(tree):
        name = prefix
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name += "." + node.name
            doc = ast.get_docstring(node)
            if doc:
                result[name] = doc
        result.update(documented_nodes(node, name))
    return result


def main():
    source_paths = ("spacr/qt/preferences.py", "spacr/qt/theme.py")
    trees = {path: ast.parse((ROOT / path).read_text()) for path in source_paths}
    prefs = trees[source_paths[0]]
    start = next(node.lineno for node in ast.walk(prefs)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Tuple)
                         and any(isinstance(part, ast.Name)
                                 and part.id == "accessibility_category"
                                 for part in target.elts)
                         for target in node.targets))
    end = next(node.lineno for node in ast.walk(prefs)
               if isinstance(node, ast.FunctionDef)
               and node.name == "_reset_to_defaults" and node.lineno > start)
    literals = {node.args[0].value for node in ast.walk(prefs)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "tr" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and start <= node.lineno < end}
    contract_path = "tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz"
    contract = json.loads(gzip.decompress((ROOT / contract_path).read_bytes()))
    added = {row["source"]: row for row in contract["added"]
             if row["table"] == "UI" and row["source"] in literals}
    assert len(added) == 10, (len(added), sorted(added))
    caption_rows = []
    proof_cache = {}
    for language in LANGUAGES:
        catalog = runpy.run_path(str(ROOT / f"spacr/qt/i18n_catalogs/{language}.py"))
        for source, arrival in sorted(added.items()):
            assert arrival["key"] == source
            assert arrival["source_sha256"] == sha(source)
            assert catalog["SOURCE_HASHES"][("UI", source)] == sha(source)
            target = catalog["UI"][source]
            assert target.strip() and target != source
            evidence = arrival["reviewed_evidence"][language]
            assert evidence["translation_sha256"] == sha(target)
            assert evidence["matching_review_records"]
            paths = []
            for proof in evidence["matching_review_records"]:
                path = proof["path"]
                if path not in proof_cache:
                    raw = (ROOT / path).read_bytes()
                    payload = json.loads(raw)
                    proof_cache[path] = raw, payload
                raw, payload = proof_cache[path]
                assert hashlib.sha256(raw).hexdigest() == proof["file_sha256"]
                assert payload["language"] == language
                records = [record for record in payload["records"]
                           if sha(json.dumps(record, ensure_ascii=False, sort_keys=True))
                           == proof["record_sha256"]]
                assert len(records) == 1
                record = records[0]
                assert record["table"] == "ui" and record["key"] == source
                assert record["source"] == source and record["source_sha256"] == sha(source)
                assert record["translation"] == target
                paths.append(path)
            caption_rows.append({"language": language, "source": source,
                                 "source_sha256": sha(source),
                                 "target_sha256": sha(target), "review_files": paths})
    arrivals_path = "tests/data/release_contracts/655_accessibility_api_arrivals_2026-10-12.json"
    arrivals = load(arrivals_path)
    assert len(arrivals) == 15
    docs = {}
    for path, tree in trees.items():
        docs.update(documented_nodes(tree, path[:-3].replace("/", ".")))
    assert {key: docs[key] for key in arrivals} == arrivals
    api_rows = []
    english = load("docs/source/_static/i18n/api/en.json")["symbols"]
    for symbol, source in arrivals.items():
        assert english[symbol]["text"] == source
        assert english[symbol]["source_sha256"] == sha(source)
    for language in LANGUAGES:
        symbols = load(f"docs/source/_static/i18n/api/{language}.json")["symbols"]
        review_path = f"docs/i18n/reviewed/api/{language}/2026-10-12-api-delta6e-000.json"
        payload = load(review_path)
        assert payload["language"] == language
        reviewed = {record["label"]: record for record in payload["records"]
                    if not record.get("retired")}
        for symbol, source in arrivals.items():
            entry = symbols[symbol]
            blocks, _layout = api.translatable_blocks(source)
            hashes = [sha(block) for block in blocks]
            assert entry["source_sha256"] == sha(source)
            assert entry["source_blocks_sha256"] == hashes
            assert entry["translation_source_blocks_sha256"] == hashes
            translated, _translated_layout = api.translatable_blocks(entry["text"])
            assert len(translated) == len(blocks)
            for index, (block, target) in enumerate(zip(blocks, translated)):
                record = reviewed[f"{symbol}#{index}"]
                assert record["source"] == block and record["source_sha256"] == sha(block)
                assert record["translation"] == target
                assert target.strip() and target != block
            api_rows.append({"language": language, "symbol": symbol,
                             "source_sha256": sha(source), "blocks": len(blocks),
                             "target_sha256": sha(entry["text"]),
                             "review_file": review_path})
    result = {
        "item": "N655", "scope": "Existing October 12 captions and API acceptance reconciliation",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "result": "PASS: historical implementation and translation debt already satisfied",
        "caption_sources": len(added), "caption_locale_checks": len(caption_rows),
        "api_symbols": len(arrivals), "api_locale_checks": len(api_rows),
        "api_reviewed_block_checks": sum(row["blocks"] for row in api_rows),
        "languages": list(LANGUAGES), "caption_rows": caption_rows, "api_rows": api_rows,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in (*source_paths, contract_path, arrivals_path)},
        "evidence_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                            for path in sorted(set(proof_cache)
                            | {row["review_file"] for row in api_rows}
                            | {f"spacr/qt/i18n_catalogs/{lang}.py" for lang in LANGUAGES}
                            | {f"docs/source/_static/i18n/api/{lang}.json"
                               for lang in ("en", *LANGUAGES)})},
        "limits": ["This verifies existing records; no new translations authored",
                   "Recorded reviews are AI technical review, not native-speaker signoff",
                   "Native screen-reader speech, Windows/macOS acceptance and Appearance tutorial refresh remain open"],
    }
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("receipt.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("result", "caption_sources", "caption_locale_checks", "api_symbols", "api_locale_checks", "api_reviewed_block_checks")}))


if __name__ == "__main__":
    main()
