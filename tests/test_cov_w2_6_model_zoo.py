"""A zoo that says what it does not know, and refuses what it cannot check.

Nothing here mocks a filesystem: the checkpoints are real files with real
torch headers (or deliberately wrong ones), the catalogues are real JSON,
and a fetch is driven over a ``file://`` URI -- which is what a lab mirror
on a NAS looks like, and is the whole reason the fetch path accepts one.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from spacr import model_zoo as mz


ZIP_HEADER = b"PK\x03\x04"
PICKLE_HEADER = b"\x80\x02"


def _checkpoint(path: Path, body: bytes = b"model bytes here",
                header: bytes = ZIP_HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("size,expected", [
    (0, mz.UNKNOWN), (None, mz.UNKNOWN), ("nonsense", mz.UNKNOWN),
    (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 ** 2, "5.0 MB"),
    (3 * 1024 ** 3, "3.0 GB"), (4096 * 1024 ** 3, "4096.0 GB"),
])
def test_a_size_is_shown_in_the_unit_a_person_reads(size, expected):
    """A model listing that says '2147483648' has told the reader nothing."""
    assert mz._human_bytes(size) == expected


def test_a_long_name_is_shortened_with_an_ellipsis():
    assert mz._shorten("abcdefghij", 5) == "abcd…"
    assert mz._shorten("abc", 5) == "abc"


# --------------------------------------------------------------------------
# checksums
# --------------------------------------------------------------------------

def test_a_checkpoint_is_hashed_in_chunks_not_read_whole(tmp_path):
    path = _checkpoint(tmp_path / "m.CP_model", body=b"x" * 5000)
    assert mz.sha256_file(path, chunk_size=64) == _digest(path)


def test_hashing_a_file_that_is_not_there_names_the_file(tmp_path):
    with pytest.raises(mz.ModelUnreadable, match="no such model file"):
        mz.sha256_file(tmp_path / "absent.CP_model")


def test_hashing_something_that_cannot_be_read_names_the_file(tmp_path):
    """A directory is the ordinary way this happens: a caller points at the
    run folder instead of the checkpoint in it."""
    (tmp_path / "folder.CP_model").mkdir()
    with pytest.raises(mz.ModelUnreadable, match="could not read"):
        mz.sha256_file(tmp_path / "folder.CP_model")


def test_a_model_that_matches_its_published_digest_verifies(tmp_path):
    path = _checkpoint(tmp_path / "m.CP_model")
    entry = mz.ModelEntry(key="m", name="m.CP_model", path=str(path),
                          sha256=_digest(path))
    assert mz.verify(entry) is True
    assert mz.verify(entry, expected="0" * 64) is False


def test_a_model_with_no_digest_is_a_caller_error_not_a_failure(tmp_path):
    """Returning False would read as 'the file is wrong' when what happened
    is 'nobody said what right looks like'."""
    path = _checkpoint(tmp_path / "m.CP_model")
    entry = mz.ModelEntry(key="m", name="m.CP_model", path=str(path))
    with pytest.raises(mz.ModelZooError, match="nothing to verify it against"):
        mz.verify(entry)


def test_verifying_a_model_that_has_not_been_downloaded_says_so():
    entry = mz.ModelEntry(key="m", name="m.CP_model", source="remote",
                          uri="https://example.org/m")
    with pytest.raises(mz.ModelUnreadable, match="has not been downloaded"):
        mz.verify(entry)


def test_verifying_a_path_that_has_gone_names_it(tmp_path):
    entry = mz.ModelEntry(key="m", name="m.CP_model",
                          path=str(tmp_path / "gone.CP_model"),
                          sha256="0" * 64)
    with pytest.raises(mz.ModelUnreadable, match="no such model file"):
        mz.verify(entry)


# --------------------------------------------------------------------------
# recognising a checkpoint
# --------------------------------------------------------------------------

def test_a_torch_header_is_recognised_in_both_save_formats(tmp_path):
    assert mz._looks_like_checkpoint(_checkpoint(tmp_path / "a.pth"))
    assert mz._looks_like_checkpoint(
        _checkpoint(tmp_path / "b.pth", header=PICKLE_HEADER))


def test_something_that_cannot_be_opened_does_not_look_like_a_checkpoint(
        tmp_path):
    (tmp_path / "folder").mkdir()
    assert mz._looks_like_checkpoint(tmp_path / "folder") is False


def test_an_html_error_page_saved_by_a_failed_download_is_refused(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"<html><body>404 Not Found</body></html>")
    with pytest.raises(mz.ModelUnreadable, match="not a PyTorch checkpoint"):
        mz.inspect_checkpoint(path)


def test_an_interrupted_download_leaves_exactly_an_empty_file(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"")
    with pytest.raises(mz.ModelUnreadable, match="empty"):
        mz.inspect_checkpoint(path)


def test_a_folder_handed_to_the_inspector_says_to_point_at_the_file(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(mz.ModelUnreadable, match="is a directory"):
        mz.inspect_checkpoint(tmp_path / "run")


def test_a_missing_checkpoint_names_the_file(tmp_path):
    with pytest.raises(mz.ModelUnreadable, match="no such model file"):
        mz.inspect_checkpoint(tmp_path / "absent.pth")


def test_a_file_that_cannot_be_read_names_the_file(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path / "m.pth")
    real_open = Path.open

    def _refuse(self, *args, **kwargs):
        if self == path:
            raise OSError(5, "Input/output error")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _refuse)
    with pytest.raises(mz.ModelUnreadable, match="could not read"):
        mz.inspect_checkpoint(path)


def test_the_shallow_check_needs_no_torch_at_all(tmp_path):
    """It is a stat and four bytes, so it can populate a list widget."""
    path = _checkpoint(tmp_path / "m.pth", body=b"y" * 100)
    out = mz.inspect_checkpoint(path)
    assert out == {"path": str(path), "size_bytes": 104,
                   "format": "zip", "loaded": False}


def test_a_deep_check_loads_the_file_and_says_it_did(tmp_path):
    path = _checkpoint(tmp_path / "m.pth")
    loaded = []
    out = mz.inspect_checkpoint(path, loader=loaded.append, deep=True)
    assert out["loaded"] is True
    assert loaded == [str(path)]
    assert mz.inspect_checkpoint(_checkpoint(tmp_path / "p.pth",
                                             header=PICKLE_HEADER),
                                 )["format"] == "pickle"


def test_a_checkpoint_of_the_wrong_architecture_says_which_kind_it_is(
        tmp_path):
    """A KeyError from inside torch names nothing the user chose and reads
    like a spaCR bug."""
    path = _checkpoint(tmp_path / "classifier.pth")

    def _explode(_path):
        raise KeyError("state_dict")

    with pytest.raises(mz.ModelUnreadable,
                       match="this is a classifier model"):
        mz.inspect_checkpoint(path, loader=_explode, deep=True)

    cellpose = _checkpoint(tmp_path / "seg.CP_model")
    with pytest.raises(mz.ModelUnreadable,
                       match="this is a Cellpose model"):
        mz.inspect_checkpoint(cellpose, loader=_explode, deep=True)


def test_a_loader_that_already_said_what_is_wrong_is_not_reworded(tmp_path):
    path = _checkpoint(tmp_path / "m.pth")

    def _explode(_path):
        raise mz.ModelUnreadable("the loader already explained this")

    with pytest.raises(mz.ModelUnreadable, match="already explained"):
        mz.inspect_checkpoint(path, loader=_explode, deep=True)


def test_the_default_loader_maps_a_checkpoint_onto_the_cpu(monkeypatch):
    """Otherwise a model saved from a GPU refuses to open on a laptop."""
    import sys
    import types

    seen = {}
    fake = types.ModuleType("torch")
    fake.load = lambda path, **kwargs: seen.update(path=path, **kwargs)
    monkeypatch.setitem(sys.modules, "torch", fake)
    mz._torch_loader("/tmp/model.pth")
    assert seen["map_location"] == "cpu"
    assert seen["weights_only"] is False


# --------------------------------------------------------------------------
# provenance beside the file
# --------------------------------------------------------------------------

def test_a_settings_csv_that_cannot_be_parsed_is_skipped(tmp_path,
                                                         monkeypatch):
    """One unreadable snapshot must not cost the model the next one."""
    model = _checkpoint(tmp_path / "run" / "m.CP_model")
    folder = tmp_path / "run" / "settings"
    folder.mkdir(parents=True, exist_ok=True)
    # Both spellings the search tries: the full filename and the stem.
    (folder / "m.CP_model.csv").write_text("this is not a key/value csv")
    (folder / "m.csv").write_text("key,value\nmodel_type,cyto3\n")

    calls = []
    real = mz._read_key_value_csv

    def _first_one_explodes(path):
        calls.append(path)
        if len(calls) == 1:
            raise ValueError("not a key/value csv")
        return real(path)

    monkeypatch.setattr(mz, "_read_key_value_csv", _first_one_explodes)
    settings, where = mz._settings_beside(model)
    assert len(calls) == 2
    assert settings["model_type"] == "cyto3"
    assert where.endswith("m.csv")


def test_who_trained_it_falls_back_through_every_name_the_run_might_use():
    assert mz._who({"author": "A. Researcher"}) == "A. Researcher"
    assert mz._who({"hostname": "gpu-box"}) == "gpu-box"
    assert mz._who({"user": "", "operator": "nan", "host": "lab"}) == "lab"
    assert mz._who({}) == mz.UNKNOWN


def test_a_setting_that_is_a_number_is_still_named_in_the_prose():
    said = mz._describe_classifier_training(
        {"classes": 4, "image_size": 224, "epochs": 30})
    assert "classes 4" in said and "224px crops" in said
    assert "30 epochs" in said


# --------------------------------------------------------------------------
# building an entry from a file
# --------------------------------------------------------------------------

def test_a_file_with_no_settings_beside_it_says_it_is_untested(tmp_path):
    path = _checkpoint(tmp_path / "cyto_v3.CP_model")
    entry = mz.entry_from_file(path)
    assert entry.kind == "cellpose"
    assert entry.trained_on == mz.UNKNOWN
    assert entry.provenance_known is False
    assert any("untested on your images" in note for note in entry.notes)
    assert entry.version == "3"
    assert entry.size_bytes == path.stat().st_size


def test_a_git_lfs_pointer_is_flagged_rather_than_listed_as_a_model(tmp_path):
    path = tmp_path / "big.CP_model"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    entry = mz.entry_from_file(path)
    assert any("Git LFS pointer" in note for note in entry.notes)


def test_a_file_that_is_not_there_cannot_become_an_entry(tmp_path):
    with pytest.raises(mz.ModelUnreadable, match="no such model file"):
        mz.entry_from_file(tmp_path / "absent.CP_model")


def test_a_file_that_vanishes_mid_scan_is_listed_with_no_size(tmp_path,
                                                              monkeypatch):
    """A model deleted while the zoo is reading it must still produce an
    entry -- an unreadable size is not a reason to drop the row."""
    path = _checkpoint(tmp_path / "m.CP_model")

    def _delete_it(candidate):
        candidate.unlink()
        return True

    monkeypatch.setattr(mz, "_looks_like_checkpoint", _delete_it)
    assert mz.entry_from_file(path).size_bytes == 0


def test_hashing_on_demand_records_the_digest_of_the_bytes_on_disk(tmp_path):
    path = _checkpoint(tmp_path / "m.CP_model")
    entry = mz.entry_from_file(path, compute_hash=True)
    assert entry.sha256 == _digest(path)
    assert entry.checksum_state == "recorded"


def test_a_digest_already_known_is_not_recomputed(tmp_path):
    path = _checkpoint(tmp_path / "m.CP_model")
    entry = mz.entry_from_file(path, compute_hash=True, sha256="a" * 64)
    assert entry.sha256 == "a" * 64


# --------------------------------------------------------------------------
# training-run metrics
# --------------------------------------------------------------------------

class _Run:
    def __init__(self, final_metrics, path="/runs/one"):
        self.final_metrics = final_metrics
        self.path = path
        self.settings = {}
        self.settings_path = ""


def test_a_held_out_score_is_reported_as_a_held_out_score():
    out = mz._metrics_from_run(_Run({
        "head": {"split": "val",
                 "best": {"accuracy": {"value": 0.91, "epoch": 7}},
                 "last": {"accuracy": {"value": 0.88, "epoch": 12}}}}))
    assert out["head best accuracy"] == "0.9100 @ epoch 7"
    assert out["head last accuracy"] == "0.8800 @ epoch 12"


def test_a_training_split_score_says_it_is_not_held_out():
    """A train-split accuracy read as a validation accuracy is how a model
    that memorised its training set gets picked."""
    out = mz._metrics_from_run(_Run({
        "head": {"split": "train",
                 "best": {"accuracy": {"value": 0.99, "epoch": 30}}}}))
    assert "not held out" in out["head best accuracy"]


def test_a_run_with_no_accuracy_reports_no_metrics():
    assert mz._metrics_from_run(_Run({"head": {"split": "train"}})) == {}
    assert mz._metrics_from_run(None) == {}


def test_a_checkpoint_finds_the_run_folder_it_sits_in(tmp_path):
    path = _checkpoint(tmp_path / "run" / "weights" / "m.pth")
    run = _Run({}, path=str(tmp_path / "run"))
    found = mz._run_for(path, {str((tmp_path / "run").resolve()): run})
    assert found is run


def test_a_checkpoint_with_no_run_anywhere_above_it_has_none(tmp_path,
                                                             monkeypatch):
    import spacr.train_compare as tc

    def _no_run(_folder):
        raise ValueError("not a training run")

    monkeypatch.setattr(tc, "load_run", _no_run)
    path = _checkpoint(tmp_path / "loose" / "m.pth")
    assert mz._run_for(path, {}) is None


def test_a_folder_that_cannot_be_scanned_costs_provenance_not_the_listing(
        tmp_path, monkeypatch):
    """A zoo that cannot list local models because one folder was unreadable
    is worse than one with thinner provenance."""
    import spacr.train_compare as tc

    def _explode(_root):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(tc, "find_runs", _explode)
    assert mz._runs_under(tmp_path) == {}


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def test_the_default_roots_are_only_the_ones_that_exist():
    for root in mz.default_local_roots():
        assert root.is_dir()


def test_a_walk_skips_hidden_folders_and_stops_at_the_limit(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "a.pth").write_bytes(ZIP_HEADER)
    for i in range(5):
        (tmp_path / f"f{i}.pth").write_bytes(ZIP_HEADER)
    found = list(mz._walk(tmp_path, max_depth=3, limit=3))
    assert len(found) == 3
    assert not any(".hidden" in str(p) for p in found)


def test_a_walk_stops_descending_at_the_depth_it_was_given(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.pth").write_bytes(ZIP_HEADER)
    (tmp_path / "a" / "shallow.pth").write_bytes(ZIP_HEADER)
    names = [p.name for p in mz._walk(tmp_path, max_depth=1, limit=100)]
    assert "shallow.pth" in names and "deep.pth" not in names


def test_a_folder_that_cannot_be_listed_is_skipped_not_fatal(tmp_path,
                                                             monkeypatch):
    bad = tmp_path / "locked"
    bad.mkdir()
    (tmp_path / "ok.pth").write_bytes(ZIP_HEADER)
    real_iterdir = Path.iterdir

    def _refuse(self):
        if self == bad:
            raise OSError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _refuse)
    assert [p.name for p in mz._walk(tmp_path, 3, 100)] == ["ok.pth"]


def test_discovery_ignores_everything_that_is_not_a_model(tmp_path):
    _checkpoint(tmp_path / "cyto.CP_model")
    _checkpoint(tmp_path / "classifier.pth")
    (tmp_path / "settings.csv").write_text("key,value\n")
    (tmp_path / "montage.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "masks.npy").write_bytes(b"\x93NUMPY")
    found = mz.discover_local(tmp_path)
    assert [e.name for e in found] == ["cyto.CP_model", "classifier.pth"]
    assert found[0].kind == "cellpose"      # Cellpose first, then by name


def test_a_single_file_can_be_discovered_without_naming_its_folder(tmp_path):
    path = _checkpoint(tmp_path / "cyto.CP_model")
    found = mz.discover_local(path)
    assert [e.name for e in found] == ["cyto.CP_model"]


def test_a_root_that_is_neither_file_nor_folder_is_skipped(tmp_path):
    assert mz.discover_local(tmp_path / "absent") == []
    assert mz.discover_local([]) == []


def test_the_same_file_reached_twice_is_listed_once(tmp_path):
    path = _checkpoint(tmp_path / "cyto.CP_model")
    found = mz.discover_local([tmp_path, path, tmp_path])
    assert len(found) == 1


def test_a_file_that_becomes_unreadable_mid_scan_is_skipped(tmp_path,
                                                            monkeypatch):
    good = _checkpoint(tmp_path / "good.CP_model")
    bad = _checkpoint(tmp_path / "bad.CP_model")
    real_resolve = Path.resolve

    def _refuse(self, *args, **kwargs):
        if self == bad:
            raise OSError(40, "Too many levels of symbolic links")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _refuse)
    assert [e.name for e in mz.discover_local(tmp_path)] == [good.name]


def test_a_file_that_disappears_between_listing_and_reading_is_skipped(
        tmp_path, monkeypatch):
    _checkpoint(tmp_path / "vanishing.CP_model")
    real = mz.entry_from_file

    def _gone(path, **kwargs):
        raise mz.ModelUnreadable(f"no such model file: {path}")

    monkeypatch.setattr(mz, "entry_from_file", _gone)
    assert mz.discover_local(tmp_path) == []


def test_a_path_under_the_bundled_pack_is_labelled_bundled(tmp_path,
                                                           monkeypatch):
    root = tmp_path / "resources" / "models"
    path = _checkpoint(root / "cyto.CP_model")
    monkeypatch.setattr(mz, "package_model_root", lambda: root)
    assert mz.discover_local(tmp_path)[0].source == "bundled"
    assert mz._under(path, root) is True
    assert mz._under(path, tmp_path / "elsewhere") is False


def test_roots_can_be_named_one_at_a_time_or_all_at_once(tmp_path):
    assert mz._as_paths(None) == []
    assert mz._as_paths(str(tmp_path)) == [tmp_path]
    assert mz._as_paths(tmp_path) == [tmp_path]
    assert mz._as_paths([tmp_path, str(tmp_path)]) == [tmp_path, tmp_path]


# --------------------------------------------------------------------------
# catalogues
# --------------------------------------------------------------------------

def test_a_catalogue_entry_needs_a_name():
    with pytest.raises(ValueError, match="at least a name"):
        mz._entry_from_mapping({"sha256": "a" * 64})


def test_an_entry_with_no_published_checksum_says_fetch_will_refuse_it():
    entry = mz._entry_from_mapping({"name": "cyto.CP_model"})
    assert entry.checksum_state == "none"
    assert any("cannot be verified" in note for note in entry.notes)
    assert entry.uri.startswith("https://huggingface.co/datasets/")


def test_a_catalogue_file_that_is_not_there_names_it(tmp_path):
    with pytest.raises(mz.ModelZooError, match="no such catalogue file"):
        mz.load_catalogue_file(tmp_path / "absent.json")


def test_a_catalogue_that_is_not_json_names_the_file(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text("{not json")
    with pytest.raises(mz.ModelZooError, match="could not read the catalogue"):
        mz.load_catalogue_file(path)


def test_a_catalogue_that_cannot_be_opened_names_the_file(tmp_path,
                                                          monkeypatch):
    path = tmp_path / "catalogue.json"
    path.write_text("[]")
    real = Path.read_text

    def _refuse(self, *args, **kwargs):
        if self == path:
            raise OSError(5, "Input/output error")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _refuse)
    with pytest.raises(mz.ModelZooError, match="could not read the catalogue"):
        mz.load_catalogue_file(path)


def test_a_catalogue_that_is_not_a_list_of_entries_says_what_it_got(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"models": {"a": 1}}))
    with pytest.raises(mz.ModelZooError, match="not a model catalogue"):
        mz.load_catalogue_file(path)


def test_an_entry_that_is_not_an_object_names_its_position(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps([{"name": "a.CP_model"}, "oops"]))
    with pytest.raises(mz.ModelZooError, match="entry 1 is a str"):
        mz.load_catalogue_file(path)


def test_an_unusable_entry_names_its_position_and_what_is_wrong(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps([{"sha256": "a" * 64}]))
    with pytest.raises(mz.ModelZooError, match="entry 0 is unusable"):
        mz.load_catalogue_file(path)


def test_a_catalogue_named_by_the_environment_joins_the_listing(tmp_path,
                                                                monkeypatch):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"models": [
        {"name": "lab_cyto.CP_model", "sha256": "b" * 64,
         "trained_on": "our own plates"}]}))
    monkeypatch.setenv(mz.CATALOGUE_ENV_VAR, str(path))
    names = [e.name for e in mz.catalogue(include_bundled=False,
                                          include_plugins=False)]
    assert "lab_cyto.CP_model" in names


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------

def test_a_path_that_exists_wins_over_a_key(tmp_path):
    """Pointing the zoo at a file you just trained has to work without
    registering it anywhere first."""
    path = _checkpoint(tmp_path / "fresh.CP_model")
    assert mz.resolve(str(path)).path == str(path.resolve())


def test_nothing_at_all_is_refused_before_anything_is_searched():
    with pytest.raises(mz.ModelZooError, match="no model given"):
        mz.resolve("")


def test_something_written_as_a_path_complains_about_the_path(tmp_path):
    """'no entry called that' would be the wrong complaint: the user pointed
    at a file and the file is not there."""
    with pytest.raises(mz.ModelUnreadable, match="no such model file"):
        mz.resolve(str(tmp_path / "absent" / "m.CP_model"))
    with pytest.raises(mz.ModelUnreadable):
        mz.resolve("~/models/m.CP_model")


def test_a_key_or_a_name_both_find_the_entry():
    pool = [mz.ModelEntry(key="cyto3", name="cyto3.CP_model")]
    assert mz.resolve("cyto3", pool) is pool[0]
    assert mz.resolve("cyto3.CP_model", pool) is pool[0]


def test_one_near_miss_is_taken_as_the_answer():
    pool = [mz.ModelEntry(key="toxo_cyto_v2", name="toxo_cyto_v2.CP_model"),
            mz.ModelEntry(key="nuclei", name="nuclei.CP_model")]
    assert mz.resolve("toxo", pool) is pool[0]


def test_several_near_misses_are_listed_rather_than_guessed_between():
    pool = [mz.ModelEntry(key="toxo_a", name="toxo_a.CP_model"),
            mz.ModelEntry(key="toxo_b", name="toxo_b.CP_model")]
    with pytest.raises(mz.ModelZooError, match="did you mean one of"):
        mz.resolve("toxo", pool)


def test_no_match_at_all_says_how_many_entries_were_searched():
    with pytest.raises(mz.ModelZooError, match="entries known"):
        mz.resolve("nothing_like_this", [mz.ModelEntry(key="a", name="a")])


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def test_a_lab_mirror_on_a_nas_is_fetched_straight_off_the_disk(tmp_path):
    source = _checkpoint(tmp_path / "mirror" / "cyto.CP_model",
                         body=b"z" * 3000)
    chunks, total = mz.open_uri(f"file://{source}", chunk_size=512)
    assert total == source.stat().st_size
    assert b"".join(chunks) == source.read_bytes()


def test_a_plain_existing_path_is_fetched_too(tmp_path):
    source = _checkpoint(tmp_path / "cyto.CP_model")
    chunks, total = mz.open_uri(str(source))
    assert b"".join(chunks) == source.read_bytes() and total > 0


def test_a_file_uri_pointing_at_nothing_names_the_file(tmp_path):
    with pytest.raises(mz.ModelUnreadable, match="no such model file"):
        mz.open_uri(f"file://{tmp_path / 'absent.CP_model'}")


def test_a_scheme_the_zoo_does_not_speak_says_what_it_expected():
    with pytest.raises(mz.ModelZooError, match="do not know how to fetch"):
        mz.open_uri("ftp://example.org/model.CP_model")


def test_an_http_uri_is_streamed_in_chunks(monkeypatch):
    """Streamed rather than read whole: a 2 GB checkpoint is not RAM."""
    import sys
    import types

    class _Response:
        headers = {"content-length": "12"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 4
            return iter([b"abcd", b"efgh", b"ijkl"])

    fake = types.ModuleType("requests")
    fake.get = lambda url, stream, timeout: _Response()
    monkeypatch.setitem(sys.modules, "requests", fake)
    chunks, total = mz.open_uri("https://example.org/m.CP_model",
                                chunk_size=4)
    assert total == 12
    assert b"".join(chunks) == b"abcdefghijkl"


def test_a_fetch_that_matches_its_checksum_installs_the_file(tmp_path):
    source = _checkpoint(tmp_path / "src" / "cyto.CP_model")
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri=f"file://{source}", sha256=_digest(source))
    landed = mz.fetch(entry, dest)
    assert landed.read_bytes() == source.read_bytes()
    assert landed.parent == dest


def test_a_second_download_of_the_same_name_does_not_destroy_the_first(
        tmp_path):
    source = _checkpoint(tmp_path / "src" / "cyto.CP_model")
    dest = tmp_path / "dest"
    dest.mkdir()
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri=f"file://{source}", sha256=_digest(source))
    first = mz.fetch(entry, dest)
    second = mz.fetch(entry, dest)
    assert first != second
    assert first.exists() and second.exists()


def test_a_name_reserved_by_a_racing_download_is_stepped_over(tmp_path,
                                                              monkeypatch):
    """`versioned_path` alone is check-then-act: two downloads finishing
    together both see the name free and the second destroys the first."""
    dest = tmp_path / "dest"
    dest.mkdir()
    temp = _checkpoint(tmp_path / "temp.part")
    real_open = os.open
    refused = {"n": 0}

    def _first_is_taken(path, flags, *args):
        if refused["n"] == 0 and flags & os.O_EXCL:
            refused["n"] += 1
            raise FileExistsError(17, "File exists")
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", _first_is_taken)
    landed = mz._claim(temp, dest, "cyto.CP_model")
    assert refused["n"] == 1
    assert landed.exists()


# --------------------------------------------------------------------------
# the bulk downloader
# --------------------------------------------------------------------------

def test_the_legacy_bulk_downloader_is_imported_late(monkeypatch):
    """`spacr.utils` pulls torch in with it, and this module stays torch-free
    at import time."""
    from spacr import utils

    assert mz._bulk_downloader() is utils.download_models


# --------------------------------------------------------------------------
# benchmarking and reporting
# --------------------------------------------------------------------------

def test_a_benchmark_needs_either_images_or_a_folder_of_them():
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model")
    with pytest.raises(ValueError, match="either images= or source="):
        mz.benchmark(entry)


def test_a_comparison_needs_either_images_or_a_folder_of_them():
    a = mz.ModelEntry(key="a", name="a.CP_model")
    b = mz.ModelEntry(key="b", name="b.CP_model")
    with pytest.raises(ValueError, match="either images= or source="):
        mz.compare_entries(a, b)


def test_a_field_set_label_lists_the_first_few_and_counts_the_rest():
    """A header naming forty fields is a header nobody reads."""
    names = [f"field_{i}" for i in range(7)]
    said = mz._fieldset_label(names, "/data/plate1")
    assert said.startswith("7 field(s) from /data/plate1: ")
    assert "… (+3)" in said
    assert mz._fieldset_label(["a", "b"], None) == "2 field(s): a, b"


def _bench_result(name="cyto.CP_model", ignored=None, notes=()):
    entry = mz.ModelEntry(key=name, name=name, trained_on="our own plates")
    return mz.BenchmarkResult(
        entry=entry, fieldset="abc123", fieldset_label="2 field(s): a, b",
        rows=[mz.FieldBenchmark("a", 12, "ok"),
              mz.FieldBenchmark("b", 9, "fail", ("empty",), "no objects")],
        seconds=1.5, honoured={"model": name, "diameter": 30},
        ignored=ignored or {}, notes=list(notes))


def test_a_benchmark_report_says_what_reached_the_model_and_what_did_not():
    """A benchmark cannot silently be a benchmark of settings nothing read."""
    result = _bench_result(ignored={"diam_mean": 30.0},
                           notes=["this model has no provenance"])
    said = mz.format_benchmarks([result])
    assert "set but ignored by Cellpose 4: diam_mean=30.0" in said
    assert "! this model has no provenance" in said
    assert "trained on: our own plates" in said
    assert "diameter 30" in said
    assert "no ground truth" in said


def test_a_benchmark_with_no_diameter_says_native_rather_than_none():
    result = _bench_result()
    result.honoured = {"model": "cyto.CP_model"}
    assert "diameter native" in mz.format_benchmarks([result])


def test_a_download_with_no_checksum_is_refused_by_default(tmp_path):
    """A download nobody can check is exactly the thing this module exists to
    stop being routine."""
    source = _checkpoint(tmp_path / "src" / "cyto.CP_model")
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri=f"file://{source}")
    with pytest.raises(mz.ModelZooError, match="no sha256 was published"):
        mz.fetch(entry, tmp_path / "dest")


def test_an_entry_with_nowhere_to_fetch_from_says_so():
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote")
    with pytest.raises(mz.ModelZooError, match="no uri to fetch it from"):
        mz.fetch(entry, "/tmp/whatever")


def test_a_keep_alive_chunk_with_no_bytes_in_it_is_skipped(tmp_path):
    """Some servers send empty chunks; counting them as progress would make
    a stalled download look alive."""
    dest = tmp_path / "dest"
    payload = ZIP_HEADER + b"real bytes"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto",
                          sha256=hashlib.sha256(payload).hexdigest())
    seen = []
    landed = mz.fetch(entry, dest,
                      opener=lambda uri: iter([b"", payload, b""]),
                      progress=lambda done, total: seen.append(done))
    assert landed.read_bytes() == payload
    assert seen == [0, len(payload)]


def test_a_download_that_arrives_empty_installs_nothing(tmp_path):
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="a" * 64)
    with pytest.raises(mz.ModelZooError, match="returned no data"):
        mz.fetch(entry, dest, opener=lambda uri: iter([]))
    assert list(dest.iterdir()) == []


def test_a_download_whose_hash_is_wrong_is_not_installed(tmp_path):
    """A checkpoint that fails this still loads and still produces masks --
    they are just not the masks the author's model produces."""
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="b" * 64)
    with pytest.raises(mz.ChecksumMismatch, match="NOT installed"):
        mz.fetch(entry, dest, opener=lambda uri: iter([ZIP_HEADER + b"x"]))
    assert list(dest.iterdir()) == []


def test_a_cancelled_download_leaves_the_destination_untouched(tmp_path):
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="c" * 64)
    with pytest.raises(mz.DownloadCancelled, match="nothing was written"):
        mz.fetch(entry, dest, cancel=lambda: True,
                 opener=lambda uri: iter([ZIP_HEADER, b"more"]))
    assert list(dest.iterdir()) == []


def test_a_dead_server_leaves_no_partial_file_at_a_model_name(tmp_path):
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="d" * 64)

    def _dies_midway(_uri):
        yield ZIP_HEADER
        raise ConnectionError("the server went away")

    with pytest.raises(ConnectionError):
        mz.fetch(entry, dest, opener=_dies_midway)
    assert list(dest.iterdir()) == []


def test_a_temporary_that_cannot_be_removed_does_not_mask_the_failure(
        tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="e" * 64)

    def _cannot_unlink(self, *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", _cannot_unlink)
    with pytest.raises(mz.ChecksumMismatch):
        mz.fetch(entry, dest, opener=lambda uri: iter([ZIP_HEADER + b"x"]))


def test_an_unverifiable_download_can_be_accepted_knowingly(tmp_path):
    dest = tmp_path / "dest"
    payload = ZIP_HEADER + b"unchecked"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto")
    landed = mz.fetch(entry, dest, require_checksum=False,
                      opener=lambda uri: (iter([payload]), len(payload)))
    assert landed.read_bytes() == payload


# --------------------------------------------------------------------------
# plugin model providers
# --------------------------------------------------------------------------

class _Contribution:
    def __init__(self, provider, key="lab.models"):
        self.provider = provider
        self.key = key


def _with_providers(monkeypatch, contributions, loader=None):
    from spacr import plugins

    diagnostics = []
    monkeypatch.setattr(plugins, "model_providers",
                        lambda: list(contributions))
    monkeypatch.setattr(plugins, "load_object",
                        loader if loader is not None else (lambda ref: ref))
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda *args: diagnostics.append(args))
    return diagnostics


def test_a_plugin_can_contribute_models_to_the_listing(monkeypatch):
    entry = mz.ModelEntry(key="lab_cyto", name="lab_cyto.CP_model",
                          source="remote", sha256="f" * 64)
    _with_providers(monkeypatch, [("lab", _Contribution(lambda: [entry]))])
    names = [e.name for e in mz.catalogue(include_bundled=False,
                                          remote=False)]
    assert "lab_cyto.CP_model" in names


def test_a_plugin_may_contribute_one_model_rather_than_a_list(monkeypatch):
    _with_providers(monkeypatch, [
        ("lab", _Contribution(lambda: {"name": "one.CP_model",
                                       "sha256": "a" * 64}))])
    names = [e.name for e in mz.catalogue(include_bundled=False,
                                          remote=False)]
    assert "one.CP_model" in names


def test_a_provider_that_is_not_callable_is_reported_not_ignored(monkeypatch):
    diagnostics = _with_providers(
        monkeypatch, [("lab", _Contribution("not.a.function"))],
        loader=lambda ref: "a string is not a provider")
    mz.catalogue(include_bundled=False, remote=False)
    assert diagnostics
    assert "is not callable" in str(diagnostics[0][2])


def test_a_provider_that_raises_is_reported_against_its_plugin(monkeypatch):
    def _explode():
        raise RuntimeError("the lab registry is down")

    diagnostics = _with_providers(monkeypatch,
                                  [("lab", _Contribution(_explode))])
    mz.catalogue(include_bundled=False, remote=False)
    assert diagnostics and diagnostics[0][0] == "lab"
    assert "lab.models" in diagnostics[0][1]


def test_a_broken_plugin_system_costs_plugins_not_the_catalogue(monkeypatch):
    """The bundled and remote entries still have to come back."""
    from spacr import plugins

    def _explode():
        raise RuntimeError("plugin registry unreadable")

    monkeypatch.setattr(plugins, "model_providers", _explode)
    entries = mz.catalogue(include_bundled=False, remote=True)
    assert isinstance(entries, list)


# --------------------------------------------------------------------------
# benchmarking argument checks
# --------------------------------------------------------------------------

class _FakeModelCompare:
    def __init__(self, names, images):
        self._names, self._images = names, images
        self.compared = []

    def load_fields(self, source, **kwargs):
        return self._names, self._images

    def compare_models(self, images, config_a, config_b, **kwargs):
        self.compared.append((config_a, config_b))
        return "report"

    class ModelConfig:
        def __init__(self, settings):
            self.settings = settings

        @classmethod
        def from_mapping(cls, settings):
            return cls(dict(settings))

        def notes(self):
            return ()


def test_a_field_name_for_every_field_or_the_rows_are_mislabelled(tmp_path,
                                                                  monkeypatch):
    import numpy as np

    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model")
    with pytest.raises(ValueError, match="field name"):
        mz.benchmark(entry, images=[np.zeros((4, 4)), np.zeros((4, 4))],
                     field_names=["only_one"])


def test_a_benchmark_with_no_field_at_all_is_refused():
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model")
    with pytest.raises(ValueError, match="no field to benchmark"):
        mz.benchmark(entry, images=[])


def test_a_comparison_loads_its_fields_from_the_folder_it_was_given(
        tmp_path, monkeypatch):
    import numpy as np

    fake = _FakeModelCompare(["a", "b"], [np.zeros((4, 4)), np.ones((4, 4))])
    monkeypatch.setattr(mz, "_model_compare", lambda: fake)
    a = mz.ModelEntry(key="a", name="a.CP_model")
    b = mz.ModelEntry(key="b", name="b.CP_model")
    assert mz.compare_entries(a, b, source=tmp_path) == "report"
    assert len(fake.compared) == 1


def test_a_close_that_fails_does_not_mask_the_failure_it_is_cleaning_up(
        tmp_path, monkeypatch):
    """The checksum mismatch is what the user has to be told about; a file
    handle that will not close on the way out is not."""
    import tempfile

    real = tempfile.NamedTemporaryFile

    class _StubbornHandle:
        def __init__(self, inner):
            self._inner = inner
            self.name = inner.name
            self._closed = 0

        def write(self, data):
            return self._inner.write(data)

        def flush(self):
            return self._inner.flush()

        def fileno(self):
            return self._inner.fileno()

        def close(self):
            self._closed += 1
            if self._closed > 1:
                raise OSError(5, "Input/output error")
            return self._inner.close()

    monkeypatch.setattr(tempfile, "NamedTemporaryFile",
                        lambda **kwargs: _StubbornHandle(real(**kwargs)))
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="a" * 64)
    with pytest.raises(mz.ChecksumMismatch):
        mz.fetch(entry, tmp_path / "dest",
                 opener=lambda uri: iter([ZIP_HEADER + b"x"]))


# ---------------------------------------------------------------------------
# sizes a person reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size,expected", [
    (1, "1 B"), (999, "999 B"),
    (1024, "1.0 KB"), (5 * 1024 ** 2, "5.0 MB"),
    (3 * 1024 ** 3, "3.0 GB"),
])
def test_a_size_is_shown_in_the_largest_unit_that_fits(size, expected):
    """Bytes have no decimals and everything above them has one.

    A weights file is the number a user compares against their disk, so "3.0
    GB" and "3221225472" are not equally useful -- and "3.2e+09 B" would be
    worse than either.
    """
    from spacr.model_zoo import _human_bytes

    assert _human_bytes(size) == expected


def test_a_size_beyond_gigabytes_stays_in_gigabytes():
    """The ladder stops at GB deliberately.

    Nothing spaCR downloads is measured in terabytes, and inventing a TB rung
    for a number that can only arrive from a corrupt manifest would make a
    wrong figure look plausible. Ten thousand gigabytes reads as obviously
    wrong, which is the useful failure.
    """
    from spacr.model_zoo import _human_bytes

    assert _human_bytes(9999 * 1024 ** 3) == "9999.0 GB"


@pytest.mark.parametrize("size", [0, -1, None, "not a number", object()])
def test_a_size_that_is_not_one_reads_as_unknown(size):
    """A manifest may carry no size, or a string, or a zero.

    ``unknown`` is the honest label; "0 B" would tell the user the download is
    free, and a raise would take down a catalogue listing over one bad row.
    """
    from spacr.model_zoo import UNKNOWN, _human_bytes

    assert _human_bytes(size) == UNKNOWN
