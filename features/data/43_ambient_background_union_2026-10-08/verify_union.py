"""Reconcile the original complete coverage with exact later source changes."""

import argparse
import ast
import difflib
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()
PATHS = (
    "spacr/qt/preferences.py",
    "spacr/qt/widgets/ambient.py",
    "spacr/qt/screens/app_screen.py",
)
AMBIENT = PATHS[1]


def _git(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", path])


def _read(name, git):
    return _git(f"HEAD:{PREFIX}/{name}") if git else (HERE / name).read_bytes()


def _source(revision, path):
    return _git(f"{revision}:{path}")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _row(report, path):
    return report["files"][path]


def _equal_lines(old, new):
    original = old.splitlines()
    current = new.splitlines()
    result = {}
    for kind, i, j, k, _ in difflib.SequenceMatcher(
        None, original, current, autojunk=False
    ).get_opcodes():
        if kind == "equal":
            result.update({i + n + 1: k + n + 1 for n in range(j - i)})
    assert all(original[a - 1] == current[b - 1] for a, b in result.items())
    return result


def _tr_literals(source):
    tree = ast.parse(source)
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tr"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def _preference_tips(source):
    tree = ast.parse(source)
    expression = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PREFERENCE_TIPS"
                for target in node.targets)
    )
    return {
        key.value: value.value
        for key, value in zip(expression.keys, expression.values)
        if isinstance(key, ast.Constant)
        and key.value in {"Animation colours", "Animation background"}
        and isinstance(value, ast.Constant)
    }


def verify(git=False, api=False, revision="HEAD"):
    receipt = json.loads(_read("receipt.json", git))
    assert receipt["target_revision"] == "237f649ad"
    current = {path: _source(revision, path) for path in PATHS}
    assert {path: _sha(data) for path, data in current.items()} == receipt[
        "target_source_sha256"
    ]
    assert _sha(_source(revision, "tests/qt/test_ambient_background_choice.py")) == receipt[
        "current_test_source_sha256"
    ]
    assert _sha(_source(revision, "tests/qt/test_fungal_growth_engine.py")) == receipt[
        "guard_test_source_sha256"
    ]

    prior = json.loads(_git("HEAD:" + receipt["prior_archive"] + "/receipt.json"))
    assert prior["modules"][AMBIENT]["remaining_lines"] == []
    assert prior["modules"][AMBIENT]["remaining_branches"] == []
    assert len(prior["modules"][PATHS[0]]["remaining_lines"]) == 84
    assert len(prior["modules"][PATHS[0]]["remaining_branches"]) == 25
    original_ambient = _source("5b655f8c7a255fe2420840d102ba5b316a69d594", AMBIENT)
    assert _sha(original_ambient) == prior["modules"][AMBIENT][
        "current_source_sha256"
    ]
    fungal = json.loads(_git("HEAD:" + receipt["fungal_archive"] + "/acceptance.json"))
    assert _sha(original_ambient) == fungal["before_sha256"]
    fungal_after = _source("fdec567f500", AMBIENT)
    assert _sha(fungal_after) == fungal["after_sha256"]
    def parent_index(source):
        assert source.count(b"        selected = set()\n") == 1
        result = source.replace(
            b"        selected = set()\n",
            b"        parents = [by_endpoint.get(edge[:2]) for edge in candidates]\n"
            b"        selected = set()\n",
            1,
        )
        assert result.count(b"                    cursor = by_endpoint.get(candidates[cursor][:2])\n") == 1
        return result.replace(
            b"                    cursor = by_endpoint.get(candidates[cursor][:2])\n",
            b"                    cursor = parents[cursor]\n",
            1,
        )

    def positive_cost_guard(source):
        target = b"        for index in reversed(range(len(candidates))):\n"
        assert source.count(target) == 1
        return source.replace(
            target,
            target
            + b"            if index not in selected and costs[index] > budget:\n"
            + b"                continue\n",
            1,
        )

    assert parent_index(original_ambient) == fungal_after
    background_ambient_gz = _read("background-measured-ambient.py.gz", git)
    assert _sha(background_ambient_gz) == receipt[
        "background_measured_ambient_archive_sha256"
    ]
    background_ambient = gzip.decompress(background_ambient_gz)
    assert _sha(background_ambient) == receipt[
        "background_measured_ambient_source_sha256"
    ]
    integrated_ambient = parent_index(background_ambient)
    assert positive_cost_guard(integrated_ambient) == current[AMBIENT]
    for name in ("guard-source-ambient.py.gz", "guard-focused.json.gz", "guard-focused.log.gz"):
        compressed = _read(name, git)
        evidence = receipt[name]
        assert _sha(compressed) == evidence["archive_sha256"]
        assert _sha(gzip.decompress(compressed)) == evidence["raw_sha256"]
    assert gzip.decompress(_read("guard-source-ambient.py.gz", git)) == current[AMBIENT]
    guard_report = json.loads(gzip.decompress(_read("guard-focused.json.gz", git)))
    assert set(guard_report["files"]) == {AMBIENT}
    assert b"18 passed" in gzip.decompress(_read("guard-focused.log.gz", git))
    guard_line = next(
        number
        for number, value in enumerate(current[AMBIENT].splitlines(), 1)
        if value == b"            if index not in selected and costs[index] > budget:"
    )
    guard_row = guard_report["files"][AMBIENT]
    assert {guard_line, guard_line + 1} <= set(guard_row["executed_lines"])
    assert {(guard_line, guard_line + 1), (guard_line, guard_line + 2)} <= {
        tuple(arc) for arc in guard_row["executed_branches"]
    }
    assert _source("5b655f8c7a255fe2420840d102ba5b316a69d594", PATHS[0]) == _source(
        "6836512a98bea77e946dd941a9b4f8a0e38068b5", PATHS[0]
    )
    assert _source("5b655f8c7a255fe2420840d102ba5b316a69d594", PATHS[2]) == _source(
        "6836512a98bea77e946dd941a9b4f8a0e38068b5", PATHS[2]
    )
    assert _source("11229945a6e803f5fa1d5ed6683d446a2b068132", AMBIENT) == original_ambient

    sources = {
        "hosted-three": {path: _source("6836512a98bea77e946dd941a9b4f8a0e38068b5", path) for path in PATHS},
        "root-focused-two": {path: _source("11229945a6e803f5fa1d5ed6683d446a2b068132", path) for path in PATHS[:2]},
        "background-86": {PATHS[0]: current[PATHS[0]], AMBIENT: background_ambient, PATHS[2]: current[PATHS[2]]},
        "integrated-7": {PATHS[0]: current[PATHS[0]], AMBIENT: integrated_ambient, PATHS[2]: current[PATHS[2]]},
        "integrated-86": {PATHS[0]: current[PATHS[0]], AMBIENT: integrated_ambient, PATHS[2]: current[PATHS[2]]},
        "fungal-55": {AMBIENT: fungal_after},
        "guard-18": {AMBIENT: current[AMBIENT]},
    }
    reports = {}
    for name in ("hosted-three", "root-focused-two", "background-86", "integrated-7"):
        compressed = _read(name + ".json.gz", git)
        assert _sha(compressed) == receipt[name]["archive_sha256"]
        report = json.loads(gzip.decompress(compressed))
        assert report["source_revision"] == receipt[name]["source_revision"]
        assert report["original_sha256"] == receipt[name]["original_sha256"]
        assert set(report["files"]) == set(sources[name])
        assert {path: _sha(data) for path, data in sources[name].items()} == receipt[name][
            "source_sha256"
        ]
        reports[name] = report
    fungal_report = json.loads(gzip.decompress(_git(
        "HEAD:" + receipt["fungal_archive"] + "/final-focused-coverage.json.gz"
    )))
    reports["fungal-55"] = fungal_report
    reports["guard-18"] = guard_report
    log = gzip.decompress(_read("integrated-7.log.gz", git))
    assert _sha(log) == receipt["integrated_log_sha256"]
    assert b"7 passed" in log
    for label in ("integrated-86.log", "integrated-86.json"):
        archived = _read(label + ".gz", git)
        key = label.replace(".", "_")
        assert _sha(archived) == receipt[key + "_archive_sha256"]
        raw = gzip.decompress(archived)
        assert _sha(raw) == receipt[key + "_sha256"]
        if label.endswith(".log"):
            assert b"86 passed" in raw
        else:
            assert set(json.loads(raw)["files"]) >= set(PATHS)
            reports["integrated-86"] = json.loads(raw)

    for path in PATHS:
        static = _row(reports["guard-18"] if path == AMBIENT else reports["integrated-7"], path)
        possible_lines = set(static["executed_lines"] + static["missing_lines"])
        possible_arcs = {tuple(arc) for arc in static["executed_branches"] + static["missing_branches"]}
        covered_lines = set()
        covered_arcs = set()
        for name, report in reports.items():
            if path not in sources[name]:
                continue
            row = _row(report, path) if name != "fungal-55" else next(
                value for key, value in report["files"].items() if key.endswith(path)
            )
            line_map = _equal_lines(sources[name][path], current[path])
            covered_lines.update(line_map[line] for line in row["executed_lines"] if line in line_map)
            covered_arcs.update(
                tuple((-1 if endpoint < 0 else 1) * line_map[abs(endpoint)] for endpoint in arc)
                for arc in row["executed_branches"]
                if all(abs(endpoint) in line_map for endpoint in arc)
            )
        missing_lines = sorted(possible_lines - covered_lines)
        missing_arcs = sorted(possible_arcs - covered_arcs)
        assert [len(missing_lines), len(missing_arcs)] == receipt["expected_remaining"][path], path
        if path in receipt["unchanged_original_allowances"]:
            maximum = receipt["unchanged_original_allowances"][path]
            assert len(missing_lines) <= maximum[0] and len(missing_arcs) <= maximum[1]
        else:
            assert [len(missing_lines), len(missing_arcs)] == receipt["app_screen_original_missing"]
        old_map = _equal_lines(_source("5b655f8c7a255fe2420840d102ba5b316a69d594", path), current[path])
        added = possible_lines - set(old_map.values())
        assert not (added - covered_lines), path
        assert not {arc for arc in possible_arcs if any(abs(end) in added for end in arc)} - covered_arcs, path
        print(path, len(missing_lines), len(missing_arcs), "added gaps 0/0")

    old_tr = _tr_literals(_source("5b655f8c7a255fe2420840d102ba5b316a69d594", PATHS[0]))
    new_tr = _tr_literals(current[PATHS[0]])
    assert new_tr - old_tr == set(receipt["source_bound_runtime_arrivals"])
    assert not old_tr - new_tr
    before_tips = _preference_tips(_source("5b655f8c7a255fe2420840d102ba5b316a69d594", PATHS[0]))
    after_tips = _preference_tips(current[PATHS[0]])
    for key, change in receipt["preference_tip_changes"].items():
        assert before_tips.get(key) == change["old"]
        assert after_tips.get(key) == change["new"]
    if api:
        sys.path.insert(0, str(ROOT / "tools"))
        import build_documentation_i18n as builder

        docs = builder.public_docstrings()
        entry = receipt["source_bound_api_arrival"]
        assert len(docs) == entry["new_count"]
        assert _sha(docs[entry["key"]].encode()) == entry["doc_sha256"]
    print("Six exact tr arrivals; one normal API arrival; no guarded ceiling changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    verify(git=args.git, api=args.api, revision=args.revision)
