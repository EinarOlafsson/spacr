"""Over-deep unknown cache metadata stays inert and cannot block relocation."""
import json
import os

import pytest

from spacr import run_journal as journal


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    home = tmp_path / "private-spacr"
    home.mkdir()
    record = home / "cache_locations.json"
    monkeypatch.setattr(journal, "_spacr_home", lambda: home)
    monkeypatch.setattr(journal, "_cache_locations_path", lambda: record)
    for key, label, variable in journal._CACHE_ROWS:
        monkeypatch.setenv(variable, "")
    return home, record


@pytest.mark.parametrize("shape", ["top-level", "unknown-value", "cache-value"])
@pytest.mark.parametrize("external", [False, True])
def test_over_deep_metadata_is_inert_and_preserves_raw_bytes_and_environment(
        metadata, monkeypatch, shape, external):
    home, record = metadata
    deep = "[" * 10000 + "0" + "]" * 10000
    payload = deep if shape == "top-level" else (
        '{"' + ("ignored" if shape == "unknown-value" else "HF_HOME") + '":' + deep + '}')
    record.write_text(payload)
    original = record.read_bytes()
    if external:
        monkeypatch.setenv("HF_HOME", str(home / "external-hf"))
    before = {variable: os.environ[variable]
              for key, label, variable in journal._CACHE_ROWS}
    assert journal._read_cache_locations() == {}
    assert journal._apply_cache_locations() == 0
    assert {variable: os.environ[variable]
            for key, label, variable in journal._CACHE_ROWS} == before
    assert record.read_bytes() == original


def test_explicit_move_recovers_from_unsupported_deep_metadata(metadata, tmp_path):
    home, record = metadata
    record.write_text('{"ignored":' + "[" * 10000 + "0" + "]" * 10000 + '}')
    source = home / "news"
    source.mkdir()
    expected = b"actual private cache contents"
    (source / "cached.bin").write_bytes(expected)
    target = journal._relocate_cache("news", tmp_path / "other-drive")
    assert target == tmp_path / "other-drive" / "spacr-news"
    assert (target / "cached.bin").read_bytes() == expected
    assert not source.exists()
    assert json.loads(record.read_text(encoding="utf-8")) == {"SPACR_NEWS_CACHE": str(target)}
    assert os.environ["SPACR_NEWS_CACHE"] == str(target)
