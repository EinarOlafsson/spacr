"""Archive and Zenodo writers given thin, odd or partly broken run folders."""
from __future__ import annotations

import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from spacr import report as rep


def _settings(folder, rows, name="a.csv"):
    (folder / "settings").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Key": list(rows), "Value": list(rows.values())}).to_csv(
        folder / "settings" / name, index=False)


def test_settings_skip_one_column_and_unreadable_tables_and_keep_values(tmp_path):
    _settings(tmp_path, {"experiment": "exp1"}, "a.csv")
    _settings(tmp_path, {"experiment": ""}, "b.csv")
    (tmp_path / "settings" / "c.csv").write_text("only\n1\n", encoding="utf-8")
    (tmp_path / "settings" / "d.csv").write_bytes(b"\x00\xff\xfe\x00garbage")
    assert rep._archive_settings(tmp_path)["experiment"] == "exp1"


@pytest.mark.parametrize("key, technology", [
    ("grna_csv", "CRISPR screen"), ("viability_plate_map", "compound screen")])
def test_the_technology_follows_the_saved_inputs(tmp_path, key, technology):
    _settings(tmp_path, {key: "/x.csv"})
    assert rep._archive_form_defaults(tmp_path)["technology"] == technology


def test_unreadable_or_unplaced_count_tables_give_no_counts(tmp_path):
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "a.db").write_bytes(b"not a database" * 100)
    with sqlite3.connect(tmp_path / "measurements" / "b.db") as con:
        pd.DataFrame({"file_name": ["notes.txt"], "cell": [3]}).to_sql(
            "pivoted_counts", con, index=False)
    assert rep._archive_counts(tmp_path) is None


def test_a_plate_map_keyed_by_well_id_is_read(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame({"plateID": ["p1"], "wellID": ["A01"], "drug": ["x"]}).to_csv(
        path, index=False)
    frame = rep._archive_plate_map(path)
    assert list(frame["Well"]) == ["A01"]


def test_the_analysis_text_survives_an_unknown_version(monkeypatch):
    import spacr.version as version

    monkeypatch.setattr(version, "get_version", lambda: "unknown")
    assert "unknown" in rep._archive_analysis_text({})




def test_an_empty_keyring_answer_falls_back_to_the_file(tmp_path, monkeypatch):
    from spacr import run_journal

    class _Ring:
        def get_password(self, service, name):
            return ""

    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: _Ring())
    monkeypatch.setattr(rep, "_zenodo_token_path", lambda: tmp_path / "t.json")
    assert rep._load_zenodo_token(sandbox=True) == ""


def test_zenodo_metadata_without_keywords_date_or_version(tmp_path, monkeypatch):
    import spacr

    monkeypatch.delattr(spacr, "__version__", raising=False)
    metadata = rep._zenodo_metadata({"title": "t", "keywords": " ; "}, tmp_path)
    assert "keywords" not in metadata and "publication_date" not in metadata
    assert "spaCR unknown" in metadata["notes"]



def test_the_zenodo_opener_never_follows_a_redirect():
    opener = rep._zenodo_opener()
    handler = next(h for h in opener.handlers
                   if isinstance(h, urllib.request.HTTPRedirectHandler))
    assert handler.redirect_request(None, None, 302, "", {}, "http://x") is None
    assert Path


def test_a_plate_map_keyed_by_prc_is_read_as_is(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame({"prc": ["plate1_r2_c2"], "drug": ["x"]}).to_csv(path, index=False)
    assert list(rep._archive_plate_map(path)["Well"]) == ["B02"]


def test_authors_without_an_affiliation_carry_none(tmp_path):
    metadata = rep._zenodo_metadata({"title": "t", "authors": "Doe Jane"},
                                    tmp_path)
    assert metadata["creators"] == [{"name": "Doe, Jane"}]



