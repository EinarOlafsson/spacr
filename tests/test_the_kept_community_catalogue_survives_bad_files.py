"""The kept community catalogue never breaks a lookup when its file is bad.

The rows the last catalogue read kept in the model cache are a convenience
for a later process: a file that cannot be written, cannot be read, or holds
a row that does not parse is logged and otherwise changes nothing.
"""
from __future__ import annotations

import json
import logging

import pytest

from spacr import model_zoo as mz

ROW = {"key": "toxoplasma_plaque_vkept", "name": "cpsam_plaque_vkept",
       "kind": "cellpose", "uri": "https://example.invalid/cpsam_plaque_vkept",
       "sha256": "0" * 64, "version": "1"}


@pytest.fixture
def kept(tmp_path, monkeypatch):
    """A private model cache whose kept catalogue file is ``.shared-catalogue.json``."""
    cache = tmp_path / "models"
    cache.mkdir()
    target = cache / ".shared-catalogue.json"
    monkeypatch.delenv("SPACR_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setattr(mz, "_shared_catalogue_file", lambda: target)
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_CACHE",
                        {"fetched_at": 0.0, "entries": ()})
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_PERSISTED",
                        {"stamp": None, "records": ()})
    return target


def test_a_row_that_cannot_be_written_leaves_no_file_and_no_part_file(kept, caplog):
    caplog.set_level(logging.DEBUG, logger=mz.LOG.name)

    mz._persist_shared_catalogue([{"key": "x", "unwritable": object()}])

    assert list(kept.parent.iterdir()) == []
    assert "could not keep the shared model catalogue" in caplog.text


def test_a_part_file_that_cannot_be_removed_is_left_quietly(kept, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=mz.LOG.name)
    removed = []

    def refuse(path):
        removed.append(path)
        raise OSError("read-only cache")

    monkeypatch.setattr(mz.os, "unlink", refuse)
    mz._persist_shared_catalogue([{"key": "x", "unwritable": object()}])

    assert len(removed) == 1 and removed[0].endswith(".part")
    assert not kept.exists()


def test_an_unreadable_kept_file_knows_no_rows(kept, caplog):
    caplog.set_level(logging.DEBUG, logger=mz.LOG.name)
    kept.write_text("{not json", encoding="utf-8")

    assert mz._persisted_shared_records() == ()
    assert mz._known_shared_entries() == ()
    assert "ignoring unreadable" in caplog.text


def test_a_kept_row_that_does_not_parse_is_skipped_and_the_rest_kept(kept, caplog):
    caplog.set_level(logging.DEBUG, logger=mz.LOG.name)
    kept.write_text(json.dumps({"models": [{"version": "1"}, ROW]}),
                    encoding="utf-8")

    assert [entry.key for entry in mz._known_shared_entries()] == [ROW["key"]]
    assert "skipping a kept shared catalogue entry" in caplog.text


def test_a_catalogue_with_no_usable_row_writes_no_kept_file(kept, tmp_path):
    served = tmp_path / "catalogue.json"
    served.write_text(json.dumps({"models": [{"version": "1"}]}),
                      encoding="utf-8")

    assert mz.shared_catalogue(uri=served.as_uri(), force=True, block=True) == ()
    assert not kept.exists()
