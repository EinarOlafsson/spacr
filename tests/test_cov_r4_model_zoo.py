"""The zoo's quiet paths: what it skips, what it de-duplicates, what it omits.

Everything already tested about ``spacr.model_zoo`` is about the loud cases --
a bad checksum, a missing file, a model with no provenance. This file pins the
other half of each of those decisions, because a listing is only trustworthy if
the warning it prints is *absent* when there is nothing to warn about:

* a settings snapshot that parses to nothing is not provenance, and the search
  carries on to the next candidate rather than stopping at an empty file;
* a checkpoint that sits outside every discovered training run falls back to
  the CSV beside it instead of borrowing another run's numbers;
* a run that recorded only the best epoch, or only the last, reports the one it
  has and does not invent the other;
* a dangling symlink is not a model;
* the same model declared twice -- by the bundled table, by a catalogue file,
  or by a plugin -- is listed once;
* a download whose scratch file was removed under it still reports the download
  failure, not a second failure from the cleanup;
* a model that says what it was trained on, and one named rather than
  downloaded, are benchmarked without a provenance warning and without opening
  a file that does not exist;
* and a clean listing prints the table and nothing else.

Nothing here reaches the network: every download runs through an injected
``opener``, and every benchmark through an injected ``segment_fn``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest

from spacr import model_zoo as mz


ZIP_HEADER = b"PK\x03\x04"


def _checkpoint(path, body: bytes = b"weights") -> str:
    """Write a file with a torch header on it; return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ZIP_HEADER + body)
    return hashlib.sha256(ZIP_HEADER + body).hexdigest()


def _settings_csv(path, mapping) -> None:
    """Write a ``Key,Value`` settings CSV the way ``save_settings`` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Key,Value"] + [f"{k},{v}" for k, v in mapping.items()]
    path.write_text("\n".join(lines) + "\n")


class _Run:
    """Stands in for a :class:`spacr.train_compare.TrainingRun`."""

    def __init__(self, settings=None, settings_path="", final_metrics=None,
                 path="/runs/one"):
        self.settings = dict(settings or {})
        self.settings_path = settings_path
        self.final_metrics = dict(final_metrics or {})
        self.path = path


def _a_field(size: int = 40, seed: int = 0) -> np.ndarray:
    image = np.full((size, size), 100.0, dtype=np.float32)
    image[4:14, 4:14] = 800.0 + seed
    image[24:34, 24:34] = 600.0 + seed
    return image


def _masks_two_objects(size: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.int32)
    mask[4:14, 4:14] = 1
    mask[24:34, 24:34] = 2
    return mask


class _Segmenter:
    """Stands in for Cellpose and records what it was asked to run."""

    def __init__(self):
        self.calls = []

    def __call__(self, images, config):
        self.calls.append((config.name, config.resolved_model, len(images)))
        return [_masks_two_objects() for _ in images]


# --------------------------------------------------------------------------
# provenance: an empty snapshot is not a snapshot
# --------------------------------------------------------------------------

def test_a_settings_file_with_nothing_in_it_is_not_the_provenance(tmp_path):
    """``<file>_settings.csv`` is looked at before ``<stem>_settings.csv``, so
    a header-only sidecar left by an interrupted ``save_settings`` would stop
    the search at a file that says nothing. It has to be stepped over."""
    model = tmp_path / "toxo.CP_model"
    _checkpoint(model)
    header_only = tmp_path / "toxo.CP_model_settings.csv"
    header_only.write_text("Key,Value\n")
    _settings_csv(tmp_path / "toxo_settings.csv",
                  {"img_src": "/data/plaques/train", "n_epochs": 25,
                   "user": "ada"})

    entry = mz.entry_from_file(model)
    assert entry.settings_path == str(tmp_path / "toxo_settings.csv")
    assert "/data/plaques/train" in entry.trained_on
    assert entry.trained_by == "ada"

    # Drive the same candidate to be present rather than empty: the nearer
    # file now wins, which is what makes the skip above a skip and not the
    # search order.
    _settings_csv(header_only, {"img_src": "/data/other/train", "user": "bob"})
    nearer = mz.entry_from_file(model)
    assert nearer.settings_path == str(header_only)
    assert "/data/other/train" in nearer.trained_on


def test_a_checkpoint_outside_every_run_uses_the_csv_beside_it(tmp_path,
                                                               monkeypatch):
    """A scan that found runs elsewhere must not lend one of them to a
    checkpoint that is not in it: the provenance shown would be another
    model's."""
    import spacr.train_compare as tc

    def _no_run(_folder):
        raise ValueError("not a training run")

    monkeypatch.setattr(tc, "load_run", _no_run)

    model = tmp_path / "run" / "weights" / "clf.pth"
    _checkpoint(model)
    _settings_csv(model.with_name("clf_settings.csv"),
                  {"src": "/data/screen2", "model_type": "resnet18",
                   "user": "grace"})

    elsewhere = {str((tmp_path / "unrelated").resolve()): _Run(
        settings={"src": "/data/screen1", "model_type": "resnet50",
                  "user": "ada"},
        settings_path="/runs/one/settings.csv",
        final_metrics={"head": {"split": "val",
                                "best": {"accuracy": {"value": 0.9,
                                                      "epoch": 3}}}})}
    missed = mz.entry_from_file(model, kind="classifier", runs=elsewhere)
    assert missed.settings_path == str(model.with_name("clf_settings.csv"))
    assert "/data/screen2" in missed.trained_on
    assert missed.trained_by == "grace"
    assert missed.metrics == {}

    # Same checkpoint, same run object, keyed on the folder it is actually in:
    # now the run's settings and its numbers are the ones that show.
    hit = mz.entry_from_file(model, kind="classifier",
                             runs={str(model.parent.resolve()):
                                   list(elsewhere.values())[0]})
    assert hit.settings_path == "/runs/one/settings.csv"
    assert "/data/screen1" in hit.trained_on
    assert hit.trained_by == "ada"
    assert hit.metrics["head best accuracy"] == "0.9000 @ epoch 3"


def test_a_run_that_recorded_one_epoch_does_not_get_the_other_invented(
        tmp_path):
    """Best and last are reported as a pair because each is misleading alone,
    but a run that only wrote one of them gets the one it wrote -- not a
    blank, and not the other one relabelled."""
    model = tmp_path / "run" / "clf.pth"
    _checkpoint(model)
    run = _Run(
        settings={"src": "/data/screen1", "model_type": "resnet50"},
        settings_path="/runs/one/settings.csv",
        final_metrics={
            "head": {"split": "val",
                     "last": {"accuracy": {"value": 0.83, "epoch": 40}}},
            "tail": {"split": "val",
                     "best": {"accuracy": {"value": 0.91, "epoch": 7}}},
        })

    entry = mz.entry_from_file(model, kind="classifier",
                               runs={str(model.parent.resolve()): run})
    assert entry.metrics["head last accuracy"] == "0.8300 @ epoch 40"
    assert entry.metrics["tail best accuracy"] == "0.9100 @ epoch 7"
    assert "head best accuracy" not in entry.metrics
    assert "tail last accuracy" not in entry.metrics


# --------------------------------------------------------------------------
# discovery: a link to nothing is not a model
# --------------------------------------------------------------------------

def test_a_link_to_a_model_that_is_gone_is_not_listed_as_a_model(tmp_path):
    """A model folder full of symlinks into a NAS is normal; a NAS that is not
    mounted turns every one of them into a name with no file behind it. Those
    must not reach the listing, and the live ones must."""
    store = tmp_path / "store"
    scan = tmp_path / "scan"
    scan.mkdir(parents=True)
    _checkpoint(store / "weights.CP_model")
    os.symlink(store / "weights.CP_model", scan / "live.CP_model")
    os.symlink(store / "not_mounted.CP_model", scan / "ghost.CP_model")

    names = [e.name for e in mz.discover_local(scan)]
    assert names == ["live.CP_model"]


# --------------------------------------------------------------------------
# the catalogue: nothing bundled, and nothing listed twice
# --------------------------------------------------------------------------

def test_a_bundled_folder_that_is_not_there_costs_only_its_own_entries(
        tmp_path, monkeypatch):
    """``resources/models`` does not exist until somebody downloads the pack.
    That is the normal state of a fresh install, not an error, and the remote
    entries still have to come back."""
    root = tmp_path / "resources" / "models"
    monkeypatch.setattr(mz, "package_model_root", lambda: root)

    before = mz.catalogue(include_bundled=True, remote=True,
                          include_plugins=False)
    assert [e for e in before if e.source == "bundled"] == []
    assert any(e.source == "remote" for e in before)

    _checkpoint(root / "pack_cyto.CP_model")
    after = mz.catalogue(include_bundled=True, remote=True,
                         include_plugins=False)
    bundled = [e for e in after if e.source == "bundled"]
    assert [e.name for e in bundled] == ["pack_cyto.CP_model"]


def test_a_model_declared_twice_in_the_table_is_listed_once(monkeypatch):
    record = {"key": "lab_cyto", "name": "lab_cyto.CP_model",
              "uri": "https://lab.example/lab_cyto", "sha256": "a" * 64}
    other = {"key": "lab_nuc", "name": "lab_nuc.CP_model",
             "uri": "https://lab.example/lab_nuc", "sha256": "b" * 64}
    monkeypatch.setattr(mz, "BUNDLED_REMOTE_MODELS",
                        (record, dict(record), other))

    names = [e.name for e in mz.catalogue(include_bundled=False, remote=True,
                                          include_plugins=False)]
    assert names.count("lab_cyto.CP_model") == 1
    assert names.count("lab_nuc.CP_model") == 1


def test_a_catalogue_file_that_repeats_an_entry_does_not_repeat_the_row(
        tmp_path):
    """A catalogue assembled by concatenating two lab lists has duplicates in
    it. The loader reports what the file says; the listing shows the model
    once."""
    record = {"key": "mirror_cyto", "name": "mirror_cyto.CP_model",
              "uri": "https://mirror.example/cyto", "sha256": "c" * 64}
    other = {"key": "mirror_nuc", "name": "mirror_nuc.CP_model",
             "uri": "https://mirror.example/nuc", "sha256": "d" * 64}
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps([record, dict(record), other]))

    assert len(mz.load_catalogue_file(path)) == 3
    names = [e.name for e in mz.catalogue(include_bundled=False, remote=True,
                                          catalogue_path=path,
                                          include_plugins=False)]
    assert names.count("mirror_cyto.CP_model") == 1
    assert names.count("mirror_nuc.CP_model") == 1


class _Contribution:
    def __init__(self, provider, key="lab.models"):
        self.provider = provider
        self.key = key


def test_a_plugin_that_offers_the_same_model_twice_adds_it_once(monkeypatch):
    from spacr import plugins

    entry = mz.ModelEntry(key="lab_cyto", name="lab_cyto.CP_model",
                          source="remote", sha256="e" * 64)
    second = mz.ModelEntry(key="lab_nuc", name="lab_nuc.CP_model",
                           source="remote", sha256="f" * 64)
    monkeypatch.setattr(plugins, "model_providers", lambda: [
        ("lab", _Contribution(lambda: [entry, replace(entry), second]))])
    monkeypatch.setattr(plugins, "load_object", lambda ref: ref)
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda *args: pytest.fail(f"provider failed: {args}"))

    names = [e.name for e in mz.catalogue(include_bundled=False, remote=False)]
    assert names.count("lab_cyto.CP_model") == 1
    assert names.count("lab_nuc.CP_model") == 1


# --------------------------------------------------------------------------
# fetch: cleaning up something that is already gone
# --------------------------------------------------------------------------

def test_a_download_whose_scratch_file_vanished_still_reports_the_download(
        tmp_path):
    """The cleanup after a failed fetch is not the story -- the failure is. A
    tmp reaper (or a user clearing the folder) that removes the ``.part`` file
    mid-transfer must not turn a ConnectionError into a FileNotFoundError from
    the unlink, and must not leave anything behind either way."""
    dest = tmp_path / "dest"
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="d" * 64)

    def _part_files():
        return [p for p in dest.iterdir() if p.name.endswith(".part")]

    def _swept_away(_uri):
        yield ZIP_HEADER
        assert _part_files(), "the download should have a scratch file by now"
        for scratch in _part_files():
            scratch.unlink()
        raise ConnectionError("the server went away")

    with pytest.raises(ConnectionError, match="the server went away"):
        mz.fetch(entry, dest, opener=_swept_away)
    assert list(dest.iterdir()) == []

    # And when the scratch file is still there, the same failure removes it --
    # which is the branch the sweep above skips.
    def _dies_midway(_uri):
        yield ZIP_HEADER
        raise ConnectionError("the server went away")

    with pytest.raises(ConnectionError, match="the server went away"):
        mz.fetch(entry, dest, opener=_dies_midway)
    assert list(dest.iterdir()) == []


# --------------------------------------------------------------------------
# benchmark: the model that needs no warning, and the model that is not a file
# --------------------------------------------------------------------------

def test_a_model_that_records_its_training_data_gets_no_warning(tmp_path):
    """The 'we do not know what this was trained on' note is the loudest thing
    in a benchmark report. It has to be missing when the model does say."""
    path = tmp_path / "toxo.CP_model"
    _checkpoint(path)
    known = mz.ModelEntry(key="toxo", name="toxo.CP_model", path=str(path),
                          trained_on="Toxoplasma plaque assay, 1120px crops",
                          trained_by="ada")

    result = mz.benchmark(known, images=[_a_field()], field_names=["f1"],
                          segment_fn=_Segmenter(), qc=False)
    assert not any("does not record what it was trained on" in note
                   for note in result.notes)

    unknown = replace(known, trained_on=mz.UNKNOWN)
    warned = mz.benchmark(unknown, images=[_a_field()], field_names=["f1"],
                          segment_fn=_Segmenter(), qc=False)
    assert any("does not record what it was trained on" in note
               for note in warned.notes)


def test_a_stock_model_named_rather_than_downloaded_is_not_opened_as_a_file(
        tmp_path):
    """A catalogue entry with no ``path`` is a model Cellpose loads by name.
    Checking it as a checkpoint would refuse every stock model there is."""
    named = mz.ModelEntry(key="cyto2", name="cyto2", source="remote",
                          uri="cellpose://cyto2",
                          trained_on="Cellpose's own training set")
    segmenter = _Segmenter()
    result = mz.benchmark(named, images=[_a_field()], field_names=["f1"],
                          segment_fn=segmenter, qc=False)
    # Passed by name, and so subject to the Cellpose 4 legacy remap the
    # config reports -- which is only possible because it was never a file.
    assert segmenter.calls == [("cyto2", "cpsam", 1)]
    assert result.rows[0].n_objects == 2
    assert any("predates Cellpose-SAM" in note for note in result.notes)

    # Give the same model a path, and the file is checked before anything
    # loads it.
    not_a_checkpoint = tmp_path / "cyto2.CP_model"
    not_a_checkpoint.write_text("<html>404</html>")
    with pytest.raises(mz.ModelUnreadable, match="cyto2.CP_model"):
        mz.benchmark(replace(named, path=str(not_a_checkpoint)),
                     images=[_a_field()], field_names=["f1"],
                     segment_fn=_Segmenter(), qc=False)


# --------------------------------------------------------------------------
# format_zoo: a clean listing is a table and nothing else
# --------------------------------------------------------------------------

def test_a_listing_with_nothing_wrong_prints_the_table_and_stops(tmp_path):
    """Three footers hang off this table -- unknown provenance, no checksum,
    and per-model notes. A listing where none of the three applies must print
    none of them, or the warnings stop meaning anything."""
    path = tmp_path / "toxo.CP_model"
    digest = _checkpoint(path)
    clean = [
        mz.ModelEntry(key="toxo", name="toxo.CP_model", path=str(path),
                      sha256=digest, verified=True,
                      trained_on="Toxoplasma plaque assay, 1120px crops",
                      trained_by="ada"),
        mz.ModelEntry(key="lab_nuc", name="lab_nuc.CP_model", source="remote",
                      uri="https://lab.example/nuc", sha256="b" * 64,
                      trained_on="HeLa nuclei, 20x", trained_by="grace"),
    ]
    listed = mz.format_zoo(clean)
    assert "toxo.CP_model" in listed and "lab_nuc.CP_model" in listed
    assert "do not record what they were trained" not in listed
    assert "no checksum on record" not in listed
    assert not [line for line in listed.splitlines()
                if line.startswith("  ! ")]

    # One model with each of the three problems, and all three footers appear.
    noisy = clean + [
        mz.ModelEntry(key="mystery", name="mystery.CP_model", path=str(path),
                      notes=("mystery.CP_model does not start with a torch "
                             "header",)),
    ]
    warned = mz.format_zoo(noisy)
    assert "do not record what they were trained" in warned
    assert "no checksum on record" in warned
    assert [line for line in warned.splitlines() if line.startswith("  ! ")]
