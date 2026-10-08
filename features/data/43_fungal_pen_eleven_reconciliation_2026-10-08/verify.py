"""Reconcile current fungal pen source with the frozen eleven-module proofs."""

import argparse
import difflib
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()
AMBIENT = "spacr/qt/widgets/ambient.py"


def _git(spec):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", spec])


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, revision="HEAD"):
    return _git(f"{revision}:{path}")


def _load_verifier(path):
    source = _read(path + "/verify_union.py")
    namespace = {"__file__": str(ROOT / path / "verify_union.py"), "__name__": "_frozen_verifier"}
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    return namespace


def _equal_lines(old, new):
    earlier = old.splitlines()
    current = new.splitlines()
    result = {}
    for kind, i, j, k, _ in difflib.SequenceMatcher(
        None, earlier, current, autojunk=False
    ).get_opcodes():
        if kind == "equal":
            result.update({i + n + 1: k + n + 1 for n in range(j - i)})
    assert all(earlier[a - 1] == current[b - 1] for a, b in result.items())
    return result


def _arc_map(arc, lines):
    if not all(abs(endpoint) in lines for endpoint in arc):
        return None
    return tuple((1 if endpoint > 0 else -1) * lines[abs(endpoint)] for endpoint in arc)


def verify(git=False, revision="HEAD", proof_revision="HEAD"):
    manifest_bytes = _read(PREFIX + "/MANIFEST.json") if git else (HERE / "MANIFEST.json").read_bytes()
    for name, expected in json.loads(manifest_bytes).items():
        data = _read(PREFIX + "/" + name) if git else (HERE / name).read_bytes()
        assert _sha(data) == expected, name
    receipt_bytes = _read(PREFIX + "/receipt.json") if git else (HERE / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["target_root_source_revision"] == "0653e111b7f8a9826081f29a972233305a76b041"
    base = receipt["prior_eleven_archive"]
    background = receipt["prior_background_archive"]
    prior_current = receipt["prior_current_archive"]
    pen = receipt["pen_archive"]
    for path, expected in receipt["frozen_receipt_sha256"].items():
        assert _sha(_read(path)) == expected, path

    _load_verifier(base)["_verify"](git=True, revision=receipt["original_target_revision"])
    _load_verifier(background)["verify"](git=True, revision=receipt["prior_source_revision"])
    old_receipt = json.loads(_read(base + "/receipt.json"))
    bg_receipt = json.loads(_read(background + "/receipt.json"))
    previous = json.loads(_read(prior_current + "/receipt.json"))

    proof_manifest = json.loads(_read(pen + "/MANIFEST.json", proof_revision))
    for name, expected in proof_manifest.items():
        data = _read(pen + "/" + name, proof_revision)
        assert len(data) == expected["bytes"], name
        assert _sha(data) == expected["sha256"], name
    acceptance = json.loads(_read(pen + "/acceptance.json", proof_revision))
    assert acceptance["before_source_sha256"] == receipt["prior_ambient_sha256"]
    assert acceptance["after_source_sha256"] == receipt["target_ambient_sha256"]
    old_source = _read(AMBIENT, receipt["prior_source_revision"])
    current_source = _read(AMBIENT, revision)
    assert _sha(old_source) == acceptance["before_source_sha256"]
    assert _sha(current_source) == acceptance["after_source_sha256"]
    assert gzip.decompress(_read(pen + "/before.py.gz", proof_revision)) == old_source
    assert gzip.decompress(_read(pen + "/after.py.gz", proof_revision)) == current_source
    contracts = json.loads(gzip.decompress(_read(pen + "/source-contracts.json.gz", proof_revision)))
    assert contracts["before_sha256"] == _sha(old_source)
    assert contracts["after_sha256"] == _sha(current_source)
    assert contracts["signature_and_prose_ast_unchanged"] is True
    assert contracts["tr_call_ast_unchanged"] is True
    assert contracts["new_control_flow"] is False
    assert b"19 passed" in gzip.decompress(_read(pen + "/focused.log.gz", proof_revision))

    previous_row = json.loads(gzip.decompress(_read(background + "/guard-focused.json.gz")))["files"][AMBIENT]
    focused = json.loads(gzip.decompress(_read(pen + "/focused-coverage.json.gz", proof_revision)))
    assert set(focused["files"]) == {AMBIENT}
    row = focused["files"][AMBIENT]
    mapping = _equal_lines(old_source, current_source)
    prior_lines = set(previous_row["executed_lines"] + previous_row["missing_lines"])
    current_lines = set(row["executed_lines"] + row["missing_lines"])
    mapped_lines = {mapping[line] for line in prior_lines if line in mapping}
    changed_lines = current_lines - mapped_lines
    assert sorted(changed_lines) == receipt["new_or_replaced_executable_lines"]
    assert sorted(changed_lines) == contracts["all_changed_statements_executed"]
    assert changed_lines <= set(row["executed_lines"])

    prior_arcs = {tuple(arc) for arc in previous_row["executed_branches"] + previous_row["missing_branches"]}
    current_arcs = {tuple(arc) for arc in row["executed_branches"] + row["missing_branches"]}
    mapped_arcs = {mapped for arc in prior_arcs if (mapped := _arc_map(arc, mapping)) is not None}
    assert current_arcs - mapped_arcs <= {tuple(arc) for arc in row["executed_branches"]}
    assert len(current_arcs - mapped_arcs) == 0
    assert len(prior_arcs) == len(current_arcs) == receipt["ambient_possible_arc_count"]

    assert set(previous["modules"]) == set(old_receipt["modules"]) == set(receipt["modules"])
    for path, expected in receipt["modules"].items():
        current_hash = _sha(_read(path, revision))
        assert current_hash == expected["source_sha256"], path
        allowance = old_receipt["modules"][path]["original_allowance"]
        assert expected["original_allowance"] == [allowance["uncovered_statements"], allowance["uncovered_branches"]]
        if path == AMBIENT:
            assert current_hash == acceptance["after_source_sha256"]
            remaining = [0, 0]
        elif path == "spacr/qt/preferences.py":
            assert current_hash == bg_receipt["target_source_sha256"][path]
            remaining = bg_receipt["expected_remaining"][path]
        else:
            assert current_hash == old_receipt["modules"][path]["current_source_sha256"]
            remaining = previous["modules"][path]["remaining"]
        assert remaining == expected["remaining"], path
        assert all(n <= ceiling for n, ceiling in zip(remaining, expected["original_allowance"])), path
        print(path, *remaining, "within unchanged allowance")
    print("Current eleven source-matched; ambient 0/0 with twelve new statements observed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--proof-revision", default="HEAD")
    arguments = parser.parse_args()
    verify(git=arguments.git, revision=arguments.revision, proof_revision=arguments.proof_revision)
