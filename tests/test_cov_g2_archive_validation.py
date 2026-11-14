"""Each way an archive package can be damaged is named by the validator."""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr import report as rep



def test_unparseable_wells_and_settings_are_skipped(tmp_path):
    assert rep._archive_well("plate1_A00_T0001F001.tif") is None
    settings = tmp_path / "settings"
    settings.mkdir()
    (settings / "gen_mask_settings.csv").write_text("only\n1\n", encoding="utf-8")
    (settings / "measure_crop_settings.csv").write_bytes(b"\xff\xfe\x00bad")
    assert rep._archive_settings(tmp_path) == {}



def test_zenodo_licences_and_empty_trees():
    assert rep._zenodo_license("") == "cc-by-4.0"
    assert rep._zenodo_license("CC0") == "cc0-1.0"
    assert rep._zenodo_zip(Path("/never.zip"), []) is None
    assert rep._zenodo_tree(Path("/no/such/folder"), "x") == []


def test_zenodo_mask_folders_fall_back_to_masks(tmp_path):
    (tmp_path / "masks").mkdir()
    assert rep._zenodo_mask_dirs(tmp_path) == [tmp_path / "masks"]
    assert rep._zenodo_mask_dirs(tmp_path / "none") == []



def test_the_zenodo_token_moves_into_a_refusing_or_willing_keyring(monkeypatch):
    from spacr import run_journal as rj

    class _Ring:
        def __init__(self, refuse):
            self.refuse, self.values = refuse, {}

        def set_password(self, service, name, value):
            if self.refuse:
                raise RuntimeError("locked")
            self.values[name] = value

        def get_password(self, service, name):
            if self.refuse:
                raise RuntimeError("locked")
            return self.values.get(name)

        def delete_password(self, service, name):
            raise RuntimeError("absent")

    monkeypatch.setattr(rj, "_notify_keyring", lambda: _Ring(refuse=True))
    assert rep._store_zenodo_token("tok-1") == "file"
    assert rep._load_zenodo_token() == "tok-1"
    willing = _Ring(refuse=False)
    monkeypatch.setattr(rj, "_notify_keyring", lambda: willing)
    assert rep._store_zenodo_token("tok-2") == "keyring"
    assert rep._load_zenodo_token() == "tok-2"
    assert rep._store_zenodo_token("") == "forgotten"


class _Opener:
    def __init__(self, outcome):
        self.outcome = outcome

    def open(self, request, timeout):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        outcome = self.outcome

        class _Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return outcome

        return _Answer()


@pytest.mark.parametrize("outcome, expected", [
    (b"", {}),
    (b"[1, 2]", {"items": [1, 2]}),
])
def test_zenodo_answers_are_decoded(monkeypatch, outcome, expected):
    monkeypatch.setattr(rep, "_zenodo_opener", lambda: _Opener(outcome))
    assert rep._zenodo_request("GET", "https://zenodo.example/api/x",
                               "tok") == expected


def test_zenodo_failures_name_the_problem_not_the_token(monkeypatch):
    import io
    import urllib.error

    monkeypatch.setattr(rep, "_zenodo_opener", lambda: _Opener(b"<html>"))
    with pytest.raises(RuntimeError, match="no JSON"):
        rep._zenodo_request("GET", "https://zenodo.example/api/x", "tok")
    refused = urllib.error.HTTPError("https://zenodo.example/api/x", 403,
                                     "Forbidden", {}, io.BytesIO(b"not json"))
    monkeypatch.setattr(rep, "_zenodo_opener", lambda: _Opener(refused))
    with pytest.raises(RuntimeError, match="HTTP 403$"):
        rep._zenodo_request("GET", "https://zenodo.example/api/x", "tok")
    monkeypatch.setattr(rep, "_zenodo_opener",
                        lambda: _Opener(urllib.error.URLError("no route")))
    with pytest.raises(RuntimeError, match="Could not reach Zenodo"):
        rep._zenodo_request("GET", "https://zenodo.example/api/x", "tok")


def test_a_deposit_needs_a_token_a_bucket_and_matching_checksums(
        monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="personal access token"):
        rep._zenodo_deposit([], {}, " ")
    monkeypatch.setattr(rep, "_zenodo_request",
                        lambda method, url, token, **k: {"id": 1})
    with pytest.raises(RuntimeError, match="no deposition with a file bucket"):
        rep._zenodo_deposit([], {}, "tok", api="https://zenodo.example/api")
    data = tmp_path / "a.zip"
    data.write_bytes(b"payload")

    def request(method, url, token, **kwargs):
        if "bucket" in url:
            return {"checksum": "md5:deadbeef"}
        return {"id": 1, "links": {"bucket": "https://zenodo.example/bucket/1"}}

    monkeypatch.setattr(rep, "_zenodo_request", request)
    with pytest.raises(RuntimeError, match="different a.zip"):
        rep._zenodo_deposit([data], {}, "tok", api="https://zenodo.example/api")
