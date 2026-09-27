"""Depositing a finished run on Zenodo, tested against a local fake API.

The run's archive package, settings, run journal, report, result tables and
masks go into one deposition with metadata from the archive form. The token
is sent only in the Authorization header, never logged or written, and kept
in the OS keyring or a mode-600 file. No test talks to Zenodo or its
sandbox: ``tests/zenodo_fake.py`` serves the API on 127.0.0.1.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import stat
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from spacr import report as rep
from tests.zenodo_fake import FakeZenodo

TOKEN = "s3cret-zenodo-token"

FORM = {"title": "Toxo screen", "description": "Two wells <test>.",
        "authors": "Doe Jane; Roe Rick", "email": "jane@example.org",
        "affiliation": "Example Lab", "microscope": "Nikon Ti2",
        "keywords": "screen; toxoplasma", "license": "CC BY 4.0",
        "release_date": "2026-09-27"}


def make_run(root: Path) -> tuple:
    src = root / "plate1"
    (src / "settings").mkdir(parents=True)
    for well in ("A01", "A02"):
        (src / f"plate1_{well}_T0001F001L01A01Z01C01.tif").write_bytes(
            well.encode() * 8)
    pd.DataFrame({"Key": ["experiment", "cell_channel"],
                  "Value": ["toxo", "1"]}).to_csv(
        src / "settings" / "measure_settings.csv", index=False)
    (src / "results").mkdir()
    pd.DataFrame({"well": ["A01"], "score": [0.5]}).to_csv(
        src / "results" / "scores.csv", index=False)
    (src / "measurements").mkdir()
    with sqlite3.connect(src / "measurements" / "measurements.db") as db:
        db.execute("create table cell (id integer)")
    (src / "masks" / "cell_mask_stack").mkdir(parents=True)
    (src / "masks" / "cell_mask_stack" / "plate1_A01.npy").write_bytes(b"m")
    run = root / "journal" / "2026-09-27_000000_abcd__measure"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(
        {"app_key": "measure", "status": "ok",
         "start_utc": "2026-09-27T00:00:00Z"}))
    (run / "settings.json").write_text(json.dumps({"src": str(src)}))
    return src, run


def _zip_names(data: bytes) -> list:
    return zipfile.ZipFile(io.BytesIO(data)).namelist()


def test_a_run_is_deposited_with_its_metadata_and_files(tmp_path, caplog):
    src, run = make_run(tmp_path)
    caplog.set_level(logging.DEBUG)
    with FakeZenodo(TOKEN) as fake:
        record = rep._zenodo_archive_run(
            src, tmp_path / "out", FORM, token=TOKEN, api=fake.api,
            include_masks=True, run_dirs=[run], search_journal=False)
        dep = fake.depositions[record["id"]]
    assert record["doi"] == "10.5072/zenodo.1"
    assert record["url"].endswith("/deposit/1")
    assert not record["published"] and record["sandbox"]
    assert sorted(dep["files"]) == sorted([
        "toxo-screen-archive.zip", "settings.zip", "run_journal.zip",
        "report.html", "results.zip", "masks.zip"])
    for entry in record["files"]:
        assert hashlib.md5(dep["files"][entry["name"]]).hexdigest() == \
            entry["md5"]
    meta = dep["metadata"]
    assert meta["upload_type"] == "dataset"
    assert meta["title"] == "Toxo screen"
    assert meta["creators"] == [
        {"name": "Doe, Jane", "affiliation": "Example Lab"},
        {"name": "Roe, Rick", "affiliation": "Example Lab"}]
    assert meta["description"] == "Two wells &lt;test&gt;."
    assert meta["license"] == "cc-by-4.0"
    assert meta["keywords"] == ["screen", "toxoplasma"]
    assert meta["publication_date"] == "2026-09-27"
    assert "measure 2026-09-27T00:00:00Z" in meta["notes"]
    assert "toxo-screen/idr/toxo-screen-study.txt" in _zip_names(
        dep["files"]["toxo-screen-archive.zip"])
    assert _zip_names(dep["files"]["results.zip"]) == [
        "measurements/measurements.db", "results/scores.csv"]
    assert _zip_names(dep["files"]["masks.zip"]) == [
        "masks/cell_mask_stack/plate1_A01.npy"]
    assert f"{run.name}/manifest.json" in _zip_names(
        dep["files"]["run_journal.zip"])
    assert b"<html" in dep["files"]["report.html"].lower()
    written = Path(record["stage"]) / "zenodo_deposit.json"
    assert json.loads(written.read_text())["doi"] == "10.5072/zenodo.1"
    assert TOKEN not in written.read_text()
    assert TOKEN not in caplog.text
    assert all(TOKEN not in r["path"] for r in fake.requests)
    assert {r["auth"] for r in fake.requests} == {f"Bearer {TOKEN}"}
    assert not (src / "zenodo_deposit.json").exists()


def test_publishing_makes_the_doi_permanent(tmp_path):
    src, run = make_run(tmp_path)
    with FakeZenodo(TOKEN) as fake:
        record = rep._zenodo_archive_run(
            src, tmp_path / "out", FORM, token=TOKEN, api=fake.api,
            publish=True, run_dirs=[run], search_journal=False)
        assert fake.depositions[1]["submitted"]
    assert record["published"] and record["doi"] == "10.5072/zenodo.1"
    assert record["url"].endswith("/records/1")
    assert "masks.zip" not in [f["name"] for f in record["files"]]


def test_a_refused_token_or_bad_metadata_is_reported_without_the_token(
        tmp_path):
    src, run = make_run(tmp_path)
    with FakeZenodo("another-token") as fake:
        with pytest.raises(RuntimeError, match="HTTP 401") as err:
            rep._zenodo_archive_run(src, tmp_path / "out", FORM, token=TOKEN,
                                    api=fake.api, run_dirs=[run],
                                    search_journal=False)
        assert TOKEN not in str(err.value)
    with FakeZenodo(TOKEN) as fake:
        with pytest.raises(RuntimeError, match="HTTP 400.*Validation"):
            rep._zenodo_archive_run(src, tmp_path / "out",
                                    dict(FORM, authors=""), token=TOKEN,
                                    api=fake.api, run_dirs=[run],
                                    search_journal=False)


def test_no_token_and_no_plain_http_to_other_hosts(tmp_path):
    src, run = make_run(tmp_path)
    with pytest.raises(ValueError, match="token is needed"):
        rep._zenodo_archive_run(src, tmp_path / "out", FORM, token="",
                                run_dirs=[run], search_journal=False)
    with pytest.raises(RuntimeError, match="Refusing") as err:
        rep._zenodo_request("POST", "http://zenodo.example.org/api", TOKEN,
                            payload={})
    assert TOKEN not in str(err.value)
    assert rep._ZENODO_API["sandbox"].startswith("https://sandbox.zenodo.org")


def test_the_token_is_kept_in_a_mode_600_file_without_a_keyring(tmp_path,
                                                                monkeypatch):
    from spacr import run_journal

    path = tmp_path / "dot" / "notification_secrets.json"
    monkeypatch.setattr(run_journal, "_notify_secrets_path", lambda: path)
    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: None)
    assert rep._store_zenodo_token(TOKEN, sandbox=True) == "file"
    token_file = rep._zenodo_token_path()
    assert token_file.parent == path.parent
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert rep._load_zenodo_token(sandbox=True) == TOKEN
    assert rep._load_zenodo_token(sandbox=False) == ""
    assert rep._store_zenodo_token("", sandbox=True) == "forgotten"
    assert not token_file.exists()


def test_the_token_goes_to_the_keyring_when_there_is_one(tmp_path,
                                                         monkeypatch):
    from spacr import run_journal

    class Ring:
        def __init__(self):
            self.kept = {}

        def set_password(self, service, name, value):
            self.kept[(service, name)] = value

        def get_password(self, service, name):
            return self.kept.get((service, name))

        def delete_password(self, service, name):
            self.kept.pop((service, name))

    ring = Ring()
    monkeypatch.setattr(run_journal, "_notify_secrets_path",
                        lambda: tmp_path / "notification_secrets.json")
    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: ring)
    assert rep._store_zenodo_token(TOKEN, sandbox=False) == "keyring"
    assert ring.kept == {("spacr-zenodo", "token"): TOKEN}
    assert not rep._zenodo_token_path().exists()
    assert rep._load_zenodo_token(sandbox=False) == TOKEN
    rep._store_zenodo_token("", sandbox=False)
    assert ring.kept == {}
