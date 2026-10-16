"""The external Toxoplasma application is ParaScape in every tracked text file.

The name it replaced is held only as hex bytes, so this file does not spell it.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORMER_BYTES = bytes.fromhex("73746172706c617374")
FORMER = re.compile(re.escape(FORMER_BYTES), re.IGNORECASE)
TEXT_SUFFIXES = {
    ".py", ".rst", ".md", ".txt", ".json", ".html", ".js", ".css", ".qss", ".svg",
    ".po", ".pot", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".tsv", ".temp",
}
_HISTORICAL_SOURCE_HASHES = {
    "features/data/ci_api_assignment_arrival_contract_2026-10-15/"
    "historical-english-surface-delta.json":
        "f3a838e9447c84a1db8b266654f046b56abd3ccb459680a77da9057c9970df76",
    "features/data/ci_api_assignment_arrival_contract_2026-10-15/"
    "twenty-arrivals-provenance.json":
        "799deffc0b58e18c44bf0a1e126d352542c8219ffd76f219368fe36ff7a92577",
    "features/data/n676_install_selective_completed_2026-10-15/"
    "export/review-evidence/56-scenes.json":
        "304e06b6b89629b3c6f438f1ffac6d11c73c2f37918afa04119eabc5750bdc6f",
}


def _current_text(name, raw):
    """Keep current prose, excluding only hash-bound historical identifiers."""
    expected = _HISTORICAL_SOURCE_HASHES.get(name)
    if expected is None:
        return raw
    assert hashlib.sha256(raw).hexdigest() == expected, (
        f"historical source evidence changed: {name}")
    data = json.loads(raw)
    if name.endswith("historical-english-surface-delta.json"):
        removed = data["removed"]
        historical = [key for key in removed if FORMER.search(key.encode())]
        assert len(historical) == 24
        data["removed"] = [key for key in removed if key not in historical]
    elif name.endswith("twenty-arrivals-provenance.json"):
        renames = data["one_for_one_renames"]
        assert sum(bool(FORMER.search(key.encode())) for key in renames) == 24
        data["one_for_one_renames"] = [
            [None if FORMER.search(key.encode()) else key, value]
            for key, value in renames.items()]
    else:
        hashes = data["capture_source"]["source_captures"][3]["exact_source"]["sha256"]
        original = "spacr/qt/" + FORMER_BYTES.decode() + ".py"
        assert original in hashes
        data["capture_source"]["source_captures"][3]["exact_source"]["sha256"] = [
            [None if key == original else key, value]
            for key, value in hashes.items()]
    return json.dumps(data, ensure_ascii=False).encode()


def _read_current_text(root, name):
    try:
        raw = (root / name).read_bytes()
    except OSError as error:
        pytest.fail(f"cannot read historical source evidence {name}: {error}")
    return _current_text(name, raw)


def _tracked():
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                           capture_output=True).stdout.split(b"\0")
    return [name.decode() for name in names if name]


def test_no_tracked_path_names_the_former_application():
    assert not [name for name in _tracked() if FORMER.search(name.encode())]


def test_no_tracked_text_names_the_former_application():
    tracked = _tracked()
    for name in _HISTORICAL_SOURCE_HASHES:
        assert name in tracked, f"missing historical source evidence: {name}"
        _read_current_text(ROOT, name)
    offenders = []
    for name in tracked:
        path = ROOT / name
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        if FORMER.search(_current_text(name, path.read_bytes())):
            offenders.append(name)
    assert not offenders


def test_a_former_name_is_found_in_any_letter_case():
    assert FORMER.search(b"Open " + FORMER_BYTES.upper())
    assert FORMER.search(b"Open " + FORMER_BYTES.capitalize())
    assert not FORMER.search(b"Open ParaScape")


@pytest.mark.parametrize("name", _HISTORICAL_SOURCE_HASHES)
def test_historical_source_fields_leave_the_other_fields_intact(name):
    raw = (ROOT / name).read_bytes()
    assert FORMER.search(raw)
    current = json.loads(_current_text(name, raw))
    original = json.loads(raw)
    assert not FORMER.search(_current_text(name, raw))
    if "one_for_one_renames" in original:
        assert [row[1] for row in current["one_for_one_renames"]] == list(
            original["one_for_one_renames"].values())
    elif "removed" in original:
        original.pop("removed")
        current.pop("removed")
        assert current == original
    else:
        original["capture_source"]["source_captures"][3]["exact_source"].pop("sha256")
        current["capture_source"]["source_captures"][3]["exact_source"].pop("sha256")
        assert current == original


@pytest.mark.parametrize("name", [
    "spacr/current.py", "docs/source/current.rst", "features/data/sibling.json",
])
def test_historical_bytes_do_not_exempt_other_paths(name):
    historical = next(iter(_HISTORICAL_SOURCE_HASHES))
    raw = (ROOT / historical).read_bytes()
    assert _current_text(name, raw) == raw
    assert FORMER.search(_current_text(name, raw))


@pytest.mark.parametrize("name", ["spacr/current.py", "docs/source/current.rst"])
def test_current_product_and_document_names_are_still_rejected(name):
    raw = b"Open " + FORMER_BYTES.upper()
    assert FORMER.search(_current_text(name, raw))


@pytest.mark.parametrize("change", ["rename", "current-product", "historical-value"])
def test_historical_drift_cannot_approve_itself(change):
    name = next(iter(_HISTORICAL_SOURCE_HASHES))
    raw = (ROOT / name).read_bytes()
    if change == "rename":
        changed = FORMER.sub(b"parascape", raw)
        assert not FORMER.search(changed)
    else:
        data = json.loads(raw)
        if change == "current-product":
            data["current_product"] = FORMER_BYTES.decode()
        else:
            data["removed"].append("new." + FORMER_BYTES.decode())
        changed = json.dumps(data).encode()
    with pytest.raises(AssertionError, match="historical source evidence changed"):
        _current_text(name, changed)


@pytest.mark.parametrize("state", ["missing", "unreadable"])
def test_historical_evidence_must_remain_readable(tmp_path, monkeypatch, state):
    name = next(iter(_HISTORICAL_SOURCE_HASHES))
    if state == "unreadable":
        def refused(_path):
            raise PermissionError("fixture read refused")
        monkeypatch.setattr(Path, "read_bytes", refused)
    with pytest.raises(pytest.fail.Exception, match="cannot read historical source evidence"):
        _read_current_text(tmp_path, name)
