"""Watch-folder helpers at their edges: odd settings, unreadable files, changing artifacts."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spacr import core


def test_numbers_default_when_blank_and_refuse_below_their_minimum():
    assert core._watch_number({"x": " "}, "x", 4.0) == 4.0
    with pytest.raises(ValueError, match="at least 1"):
        core._watch_number({"x": 0.5}, "x", 4.0, minimum=1.0)


def test_raw_channel_positions_require_a_source_origin():
    assert core._watch_source_channels({"channels": "[0, 3]"}) == {
        "1", "2", "3", "4"}
    with pytest.raises(ValueError, match="distinct non-negative"):
        core._watch_source_channels({"channels": "[0, 1"})
    with pytest.raises(ValueError, match="conversion_map.csv"):
        core._watch_source_channels({"channels": [0], "metadata_type": "custom"})


def test_an_unusable_filename_pattern_is_cached_as_none():
    cache = {}
    settings = {"metadata_type": "custom", "custom_regex": "([unclosed"}
    assert core._watch_pattern(settings, ".tif", cache) is None
    assert cache[".tif"] is None


def test_unreadable_files_are_named_by_kind(tmp_path):
    from PIL import Image

    good = tmp_path / "a.png"
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(good)
    assert core._watch_unreadable(str(good)) is None
    broken = tmp_path / "b.png"
    broken.write_bytes(b"\x89PNG truncated")
    assert core._watch_unreadable(str(broken))
    other = tmp_path / "c.nd2"
    other.write_bytes(b"data")
    assert core._watch_unreadable(str(other)) is None
    empty = tmp_path / "d.nd2"
    empty.write_bytes(b"")
    assert core._watch_unreadable(str(empty)) is None


def test_an_unreadable_ledger_is_set_aside_and_restarted(tmp_path, capsys):
    path = tmp_path / "watch_ledger.json"
    path.write_text("{broken", encoding="utf-8")
    ledger = core._watch_load_ledger(str(path), str(tmp_path))
    assert isinstance(ledger, dict)
    assert (tmp_path / "watch_ledger.json.unreadable").exists()
    assert "could not be read" in capsys.readouterr().out


def test_a_link_across_devices_falls_back_to_a_copy(tmp_path, monkeypatch):
    source = tmp_path / "a.tif"
    source.write_bytes(b"x")

    def refuse(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr(core.os, "link", refuse)
    core._watch_link(str(source), str(tmp_path / "b.tif"))
    assert (tmp_path / "b.tif").read_bytes() == b"x"


def test_a_measure_recipe_with_non_finite_values_is_refused():
    with pytest.raises(ValueError, match="finite"):
        core._watch_measure_recipe({"channels": [float("nan")]})


def test_a_vanished_database_snapshot_is_refused(tmp_path):
    with pytest.raises(ValueError, match="snapshot disappeared"):
        core._watch_collect(str(tmp_path), str(tmp_path / "work"), "k",
                            database_snapshot=str(tmp_path / "gone.db"))


def test_an_artifact_that_cannot_be_opened_or_changes_is_refused(
        tmp_path, monkeypatch):
    path = tmp_path / "a.csv"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="missing or unsafe"):
        core._watch_artifact_sha256(str(tmp_path / "none.csv"))

    real_open = os.open

    def refuse(target, flags, *a):
        if str(target) == str(path):
            raise OSError("busy")
        return real_open(target, flags, *a)

    monkeypatch.setattr(core.os, "open", refuse)
    with pytest.raises(ValueError, match="cannot be opened"):
        core._watch_artifact_sha256(str(path))
    monkeypatch.setattr(core.os, "open", real_open)
    identity = core._watch_file_identity(str(path))
    monkeypatch.setattr(core, "_watch_file_identity",
                        lambda p: [identity[0], identity[1], identity[2] - 1,
                                   identity[3], identity[4]])
    with pytest.raises(ValueError, match="changed before hashing"):
        core._watch_artifact_sha256(str(path))


def test_a_database_that_is_not_a_regular_file_is_refused(tmp_path):
    folder = tmp_path / "measurements" / "measurements.db"
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match="not a regular file"):
        core._watch_snapshot_database(str(tmp_path))


def test_settings_name_one_existing_folder(tmp_path):
    assert core._watch_check_settings({"src": [str(tmp_path)]})[0] == str(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        core._watch_check_settings({"src": str(tmp_path / "none")})


def test_a_snapshot_of_a_changed_source_is_not_copied(tmp_path):
    source = tmp_path / "a.db"
    source.write_bytes(b"x")
    assert core._watch_copy_snapshot(str(source), str(tmp_path / "b.db"),
                                     None) is None


def test_a_conversion_map_with_repeated_or_zero_identifiers_is_refused(
        tmp_path):
    header = "target,source,plate,well,field,channel,z,t,status\n"
    row = "plate1_A01_T0001F001L01A01Z01C01.tif,/raw/a.tif,plate1,A01,1,1,1,1,converted\n"
    (tmp_path / "conversion_map.csv").write_text(
        header.replace("status", "status,status") + row.replace(
            "converted", "converted,converted"), encoding="utf-8")
    with pytest.raises(ValueError):
        core._watch_map_manifest(str(tmp_path), {})
    (tmp_path / "conversion_map.csv").write_text(
        header + row.replace(",1,1,1,1,", ",0,1,1,1,"), encoding="utf-8")
    with pytest.raises(ValueError):
        core._watch_map_manifest(str(tmp_path), {})


def test_legacy_ledger_keys_migrate_without_accepting_changed_or_missing_sources(tmp_path):
    """Portable spelling must not turn a changed acquisition into a completed field."""
    folder = tmp_path / 'nested'
    folder.mkdir()
    unchanged = folder / 'same.tif'
    changed = folder / 'changed.tif'
    unchanged.write_bytes(b'original unchanged camera pixels')
    changed.write_bytes(b'new camera pixels')
    same_stat, changed_stat = unchanged.stat(), changed.stat()
    same = [same_stat.st_dev, same_stat.st_ino, same_stat.st_size,
            same_stat.st_mtime_ns, same_stat.st_ctime_ns]
    stale = [changed_stat.st_dev, changed_stat.st_ino, changed_stat.st_size + 1,
             changed_stat.st_mtime_ns, changed_stat.st_ctime_ns]
    entry = {'files': {'nested\\same.tif': [1, 2]},
             'snapshot_sha256': {'nested\\same.tif': 'kept-digest'},
             'source_identity': {'nested\\same.tif': same.copy(),
                                 'nested\\changed.tif': stale.copy(),
                                 'nested\\missing.tif': same.copy(),
                                 'not-an-identity': 'preserved'},
             'collection_checkpoint': {'files': {'nested\\same.tif': 'kept'},
                                       'unchanged': None}}
    before_bytes = unchanged.read_bytes(), changed.read_bytes()
    core._watch_migrate_windows_entry(entry, str(tmp_path))
    assert entry['files'] == {'nested/same.tif': [1, 2]}
    assert entry['snapshot_sha256'] == {'nested/same.tif': 'kept-digest'}
    assert entry['collection_checkpoint'] == {'files': {'nested/same.tif': 'kept'},
                                              'unchanged': None}
    assert entry['source_identity']['nested/same.tif'] == core._watch_file_identity(unchanged)
    assert entry['source_identity']['nested/changed.tif'] == stale
    assert entry['source_identity']['nested/missing.tif'] == same
    assert entry['source_identity']['not-an-identity'] == 'preserved'
    assert (unchanged.read_bytes(), changed.read_bytes()) == before_bytes
    no_identity = {'files': ['an older nonmapping record']}
    core._watch_migrate_windows_entry(no_identity, str(tmp_path))
    assert no_identity == {'files': ['an older nonmapping record']}


def test_windows_ledger_loader_migrates_only_legacy_regular_file_identities(tmp_path, monkeypatch):
    """The Windows adapter updates legacy change-time keys without trusting unsafe files."""
    image = tmp_path / 'same.tif'
    image.write_bytes(b'acquired image')
    directory = tmp_path / 'folder'
    directory.mkdir()
    info = image.stat()
    identity = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    folder_info = directory.stat()
    folder_identity = [folder_info.st_dev, folder_info.st_ino, folder_info.st_size,
                       folder_info.st_mtime_ns, folder_info.st_ctime_ns]
    changed = tmp_path / 'changed.tif'
    changed.write_bytes(b'changed acquisition')
    changed_info = changed.stat()
    nonlegacy = [changed_info.st_dev, changed_info.st_ino, changed_info.st_size,
                 changed_info.st_mtime_ns, changed_info.st_ctime_ns - 1]
    entry = {'files': {4: 'nonpath metadata', 'same.tif': 'original'},
             'source_identity': {'same.tif': identity, 'folder': folder_identity,
                                 'changed.tif': nonlegacy}}
    record = {'fields': {'valid': entry, 'legacy_nonmapping': None}}
    path = tmp_path / 'watch_ledger.json'
    path.write_text(json.dumps(record))
    before = path.read_bytes(), image.read_bytes()

    class WindowsOS:
        name = 'nt'

        def __getattr__(self, name):
            return getattr(os, name)

    with monkeypatch.context() as patch:
        patch.setattr(core, 'os', WindowsOS())
        patch.setattr(core, '_watch_windows_change_ns', lambda **kwargs: 123456789)
        loaded = core._watch_load_ledger(str(path), str(tmp_path))
    actual = loaded['fields']['valid']
    assert actual['source_identity']['same.tif'] == identity[:4] + [123456789]
    assert actual['source_identity']['folder'] == folder_identity
    assert actual['source_identity']['changed.tif'] == nonlegacy
    assert actual['files'] == {'4': 'nonpath metadata', 'same.tif': 'original'}
    assert loaded['fields']['legacy_nonmapping'] is None
    assert (path.read_bytes(), image.read_bytes()) == before
    assert core._watch_portable_keys({4: 'metadata', 'a\\b': 'path'}) == {4: 'metadata', 'a/b': 'path'}
