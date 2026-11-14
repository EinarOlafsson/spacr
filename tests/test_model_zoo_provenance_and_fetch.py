"""Table-driven tests of the model zoo's provenance, listing, download and Timepose paths.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: :func:`~spacr.model_zoo.entry_from_file` (which
settings snapshot or training run is the provenance), :func:`discover_local`,
:func:`catalogue` with :func:`load_catalogue_file` (nothing listed twice,
nothing bundled is fine), :func:`fetch`, :func:`benchmark`,
:func:`format_zoo`, :func:`_human_bytes`, and the Timepose release helpers
:func:`_timepose_files` and :func:`_load_timepose`.  Nothing reaches the
network: downloads run through injected openers/downloaders and benchmarks
through an injected ``segment_fn``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Run:
    """Stands in for a :class:`spacr.train_compare.TrainingRun`."""

    def __init__(self, settings, settings_path, final_metrics):
        self.settings = dict(settings)
        self.settings_path = settings_path
        self.final_metrics = dict(final_metrics)


class _Segmenter:
    """Stands in for Cellpose: two square objects per field, calls recorded."""

    def __init__(self):
        self.calls = []

    def __call__(self, images, config):
        self.calls.append((config.name, config.resolved_model, len(images)))
        masks = []
        for _ in images:
            mask = np.zeros((40, 40), dtype=np.int32)
            mask[4:14, 4:14] = 1
            mask[24:34, 24:34] = 2
            masks.append(mask)
        return masks


def _a_field() -> np.ndarray:
    image = np.full((40, 40), 100.0, dtype=np.float32)
    image[4:14, 4:14] = 800.0
    image[24:34, 24:34] = 600.0
    return image


NO_PROVENANCE = ("no settings snapshot found beside this model, so what it was "
                 "trained on is unknown — treat it as untested on your images")


# ---------------------------------------------------------------------------
# entry_from_file
# ---------------------------------------------------------------------------

def test_entry_from_file_takes_provenance_from_the_nearest_real_record(subtests, tmp_path,
                                                                       monkeypatch):
    """Empty sidecars are skipped, only the checkpoint's own run lends numbers, nothing is invented."""
    import spacr.train_compare as tc

    def _no_run(_folder):
        raise ValueError("not a training run")

    monkeypatch.setattr(tc, "load_run", _no_run)

    toxo = tmp_path / "cp" / "toxo.CP_model"
    _checkpoint(toxo)
    sidecar = toxo.with_name("toxo.CP_model_settings.csv")
    sidecar.write_text("Key,Value\n", encoding="utf-8")
    stem_csv = toxo.with_name("toxo_settings.csv")
    _settings_csv(stem_csv, {"img_src": "/data/plaques/train", "n_epochs": 25,
                             "user": "ada"})

    clf = tmp_path / "run" / "weights" / "clf.pth"
    _checkpoint(clf)
    beside = clf.with_name("clf_settings.csv")
    _settings_csv(beside, {"src": "/data/screen2", "model_type": "resnet18",
                           "user": "grace"})
    best_run = _Run({"src": "/data/screen1", "model_type": "resnet50", "user": "ada"},
                    "/runs/one/settings.csv",
                    {"head": {"split": "val",
                              "best": {"accuracy": {"value": 0.9, "epoch": 3}}}})
    split_run = _Run({"src": "/data/screen1", "model_type": "resnet50"},
                     "/runs/one/settings.csv",
                     {"head": {"split": "val",
                               "last": {"accuracy": {"value": 0.83, "epoch": 40}}},
                      "tail": {"split": "val",
                               "best": {"accuracy": {"value": 0.91, "epoch": 7}}}})
    train_run = _Run({"src": "/data/screen1"}, "/runs/two/settings.csv",
                     {"head": {"split": "train",
                               "best": {"accuracy": {"value": 0.5, "epoch": 1}}}})
    bare = tmp_path / "bare" / "mystery.pth"
    bare.parent.mkdir()
    bare.write_bytes(b"<html>404</html>")
    here = str(clf.parent.resolve())

    cases = [
        # (case id, path, kwargs, before(), expected fields)
        ("a header-only sidecar is stepped over", toxo, {}, None,
         dict(kind="cellpose", settings_path=str(stem_csv),
              trained_on="/data/plaques/train, 25 epochs", trained_by="ada",
              metrics={}, notes=())),
        ("a filled sidecar is nearer and wins", toxo, {},
         lambda: _settings_csv(sidecar, {"img_src": "/data/other/train", "user": "bob"}),
         dict(kind="cellpose", settings_path=str(sidecar),
              trained_on="/data/other/train", trained_by="bob", metrics={}, notes=())),
        ("a run elsewhere lends nothing; the csv beside is used", clf,
         {"kind": "classifier",
          "runs": {str((tmp_path / "unrelated").resolve()): best_run}}, None,
         dict(kind="classifier", settings_path=str(beside),
              trained_on="/data/screen2, resnet18", trained_by="grace",
              metrics={}, notes=())),
        ("the checkpoint's own run gives settings and numbers", clf,
         {"kind": "classifier", "runs": {here: best_run}}, None,
         dict(settings_path="/runs/one/settings.csv",
              trained_on="/data/screen1, resnet50", trained_by="ada",
              metrics={"head best accuracy": "0.9000 @ epoch 3"}, notes=())),
        ("a run that recorded one epoch per head gets only that one", clf,
         {"kind": "classifier", "runs": {here: split_run}}, None,
         dict(metrics={"head last accuracy": "0.8300 @ epoch 40",
                       "tail best accuracy": "0.9100 @ epoch 7"},
              trained_by=mz.UNKNOWN)),
        ("a train-split best is labelled as not held out", clf,
         {"kind": "classifier", "runs": {here: train_run}}, None,
         dict(metrics={"head best accuracy":
                       "0.5000 @ epoch 1 (train split — not held out)"},
              trained_on="/data/screen1")),
        ("no snapshot and no torch header are both noted", bare, {}, None,
         dict(kind="classifier", settings_path="", trained_on=mz.UNKNOWN,
              trained_by=mz.UNKNOWN, notes=(
                  NO_PROVENANCE,
                  "mystery.pth does not start with a torch header; it may be a "
                  "Git LFS pointer, a failed download, or not a model at all"))),
    ]
    for case_id, path, kwargs, before, expected in cases:
        with subtests.test(msg=case_id):
            if before is not None:
                before()
            entry = mz.entry_from_file(path, **kwargs)
            assert {k: getattr(entry, k) for k in expected} == expected
            assert (entry.name, entry.path, entry.source) == (
                path.name, str(path.resolve()), "local")
            assert entry.size_bytes == path.stat().st_size

    with subtests.test(msg="a path that is not a file is unreadable"):
        missing = tmp_path / "gone.CP_model"
        with pytest.raises(mz.ModelUnreadable) as caught:
            mz.entry_from_file(missing)
        assert str(caught.value) == f"no such model file: {missing}"


# ---------------------------------------------------------------------------
# discover_local
# ---------------------------------------------------------------------------

@pytest.mark.needs_symlink
def test_discover_local_lists_only_model_files_that_are_there(subtests, tmp_path):
    """Dangling links and non-model files are skipped; Cellpose models sort first."""
    store = tmp_path / "store"
    scan = tmp_path / "scan"
    scan.mkdir(parents=True)
    _checkpoint(store / "weights.CP_model")
    os.symlink(store / "weights.CP_model", scan / "live.CP_model")
    os.symlink(store / "not_mounted.CP_model", scan / "ghost.CP_model")
    _checkpoint(scan / "a_classifier.pth")
    (scan / "notes.csv").write_text("Key,Value\n", encoding="utf-8")
    cases = [
        # (case id, roots, expected (name, kind) list)
        ("a folder with a dangling link", scan,
         [("live.CP_model", "cellpose"), ("a_classifier.pth", "classifier")]),
        ("a dangling link named directly", scan / "ghost.CP_model", []),
        ("a live link named directly", [scan / "live.CP_model"],
         [("live.CP_model", "cellpose")]),
        ("a root that does not exist", tmp_path / "nowhere", []),
    ]
    for case_id, roots, expected in cases:
        with subtests.test(msg=case_id):
            assert [(e.name, e.kind) for e in mz.discover_local(roots)] == expected


# ---------------------------------------------------------------------------
# catalogue + load_catalogue_file
# ---------------------------------------------------------------------------

class _Contribution:
    def __init__(self, provider, key="lab.models"):
        self.provider = provider
        self.key = key


def test_catalogue_lists_each_model_once_from_every_source(subtests, tmp_path, monkeypatch):
    """Bundled, table, catalogue-file and plugin entries are each listed once; a missing pack is fine."""
    from spacr import plugins

    def record(key, letter, host="lab.example"):
        return {"key": key, "name": f"{key}.CP_model",
                "uri": f"https://{host}/{key}", "sha256": letter * 64}

    cyto, nuc = record("lab_cyto", "a"), record("lab_nuc", "b")
    mirror_cyto = record("mirror_cyto", "c", "mirror.example")
    mirror_nuc = record("mirror_nuc", "d", "mirror.example")
    cat_file = tmp_path / "catalogue.json"
    cat_file.write_text(json.dumps([mirror_cyto, dict(mirror_cyto), mirror_nuc]),
                        encoding="utf-8")
    root = tmp_path / "resources" / "models"
    plugin_cyto = mz.ModelEntry(key="plug_cyto", name="plug_cyto.CP_model",
                                source="remote", sha256="e" * 64)
    plugin_nuc = mz.ModelEntry(key="plug_nuc", name="plug_nuc.CP_model",
                               source="remote", sha256="f" * 64)

    def table_twice(mp):
        mp.setattr(mz, "BUNDLED_REMOTE_MODELS", (cyto, dict(cyto), nuc))

    def plugin_twice(mp):
        mp.setattr(plugins, "model_providers", lambda: [
            ("lab", _Contribution(lambda: [plugin_cyto, replace(plugin_cyto),
                                           plugin_nuc]))])
        mp.setattr(plugins, "load_object", lambda ref: ref)
        mp.setattr(plugins, "record_diagnostic",
                   lambda *args: pytest.fail(f"provider failed: {args}"))

    cases = [
        # (case id, patch, catalogue kwargs, names each listed exactly once)
        ("the bundled table repeats a record", table_twice,
         dict(include_bundled=False, remote=True, include_plugins=False),
         ["lab_cyto.CP_model", "lab_nuc.CP_model"]),
        ("a catalogue file repeats a record", None,
         dict(include_bundled=False, remote=True, catalogue_path=cat_file,
              include_plugins=False),
         ["mirror_cyto.CP_model", "mirror_nuc.CP_model"]),
        ("a plugin offers a model twice", plugin_twice,
         dict(include_bundled=False, remote=False),
         ["plug_cyto.CP_model", "plug_nuc.CP_model"]),
    ]
    for case_id, patch, kwargs, once in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            if patch is not None:
                patch(mp)
            names = [e.name for e in mz.catalogue(**kwargs)]
            assert {n: names.count(n) for n in once} == {n: 1 for n in once}

    with subtests.test(msg="the loader reports the file as written"):
        loaded = mz.load_catalogue_file(cat_file)
        assert [(e.key, e.name, e.uri, e.sha256) for e in loaded] == [
            (r["key"], r["name"], r["uri"], r["sha256"])
            for r in (mirror_cyto, mirror_cyto, mirror_nuc)]

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "scalar.json").write_text("3", encoding="utf-8")
    (bad / "row.json").write_text('{"models": [1]}', encoding="utf-8")
    (bad / "broken.json").write_text("{", encoding="utf-8")
    refusals = [
        ("missing file", bad / "none.json", f"no such catalogue file: {bad / 'none.json'}"),
        ("not a list", bad / "scalar.json",
         f"{bad / 'scalar.json'} is not a model catalogue — expected a list of "
         f"entries, or an object with a 'models' list, got int"),
        ("a row that is not an object", bad / "row.json",
         f"{bad / 'row.json'}: entry 0 is a int, not an object"),
        ("unparseable json", bad / "broken.json",
         f"could not read the catalogue {bad / 'broken.json'}: "),
    ]
    for case_id, path, message in refusals:
        with subtests.test(msg=f"load_catalogue_file refuses: {case_id}"):
            with pytest.raises(mz.ModelZooError) as caught:
                mz.load_catalogue_file(path)
            assert str(caught.value).startswith(message)

    with subtests.test(msg="a pack folder that is not there costs only its entries"), \
            monkeypatch.context() as mp:
        mp.setattr(mz, "package_model_root", lambda: root)
        before = mz.catalogue(include_bundled=True, remote=True, include_plugins=False)
        assert [e for e in before if e.source == "bundled"] == []
        assert [e for e in before if e.source == "remote"] != []
        _checkpoint(root / "pack_cyto.CP_model")
        after = mz.catalogue(include_bundled=True, remote=True, include_plugins=False)
        assert [(e.name, e.source) for e in after if e.source == "bundled"] == [
            ("pack_cyto.CP_model", "bundled")]
        assert [e.name for e in after if e.source != "bundled"] == [
            e.name for e in before]


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@pytest.mark.posix_only
def test_fetch_leaves_the_destination_clean_on_every_failure(subtests, tmp_path):
    """A failed download raises its own error and leaves no scratch file, even one already gone."""
    entry = mz.ModelEntry(key="cyto", name="cyto.CP_model", source="remote",
                          uri="mirror://cyto", sha256="d" * 64)

    def swept_away(dest):
        def opener(_uri):
            yield ZIP_HEADER
            parts = [p for p in dest.iterdir() if p.name.endswith(".part")]
            assert len(parts) == 1, "the download should have one scratch file by now"
            parts[0].unlink()
            raise ConnectionError("the server went away")
        return opener

    def dies_midway(_dest):
        def opener(_uri):
            yield ZIP_HEADER
            raise ConnectionError("the server went away")
        return opener

    cases = [
        # (case id, entry, opener factory, exception type, exact message prefix)
        ("scratch file swept away mid-transfer", entry, swept_away, ConnectionError,
         "the server went away"),
        ("server dies mid-transfer", entry, dies_midway, ConnectionError,
         "the server went away"),
        ("no uri", replace(entry, uri=""), dies_midway, mz.ModelZooError,
         "cyto.CP_model has no uri to fetch it from (source=remote)"),
        ("no published checksum", replace(entry, sha256=""), dies_midway,
         mz.ModelZooError, "refusing to install cyto.CP_model: no sha256 was published"),
    ]
    for index, (case_id, item, factory, exc, message) in enumerate(cases):
        with subtests.test(msg=case_id):
            dest = tmp_path / f"dest{index}"
            dest.mkdir()
            with pytest.raises(exc) as caught:
                mz.fetch(item, dest, opener=factory(dest))
            assert str(caught.value).startswith(message)
            assert list(dest.iterdir()) == []


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def test_benchmark_warns_only_about_unknown_provenance_and_checks_only_real_files(
        subtests, tmp_path):
    """A model that says what it saw gets no warning; a named stock model is never opened as a file."""
    path = tmp_path / "toxo.CP_model"
    _checkpoint(path)
    known = mz.ModelEntry(key="toxo", name="toxo.CP_model", path=str(path),
                          trained_on="Toxoplasma plaque assay, 1120px crops",
                          trained_by="ada")
    named = mz.ModelEntry(key="cyto2", name="cyto2", source="remote",
                          uri="cellpose://cyto2", trained_on="Cellpose's own training set")
    warning = ("toxo.CP_model does not record what it was trained on, so a good "
               "score here says it works on these fields and nothing more.")
    cases = [
        # (case id, entry, expected segmenter calls, warning present)
        ("a model with recorded training data", known,
         [("toxo.CP_model", str(path), 1)], False),
        ("the same model without it", replace(known, trained_on=mz.UNKNOWN),
         [("toxo.CP_model", str(path), 1)], True),
        ("a stock model passed by name and remapped", named, [("cyto2", "cpsam", 1)],
         False),
    ]
    for case_id, entry, calls, warned in cases:
        with subtests.test(msg=case_id):
            segmenter = _Segmenter()
            result = mz.benchmark(entry, images=[_a_field()], field_names=["f1"],
                                  segment_fn=segmenter, qc=False)
            if entry is not named:
                assert segmenter.calls == calls
            else:
                assert segmenter.calls == calls
                assert [n for n in result.notes if "predates Cellpose-SAM" in n] != []
            assert [(r.field, r.n_objects, r.severity) for r in result.rows] == [
                ("f1", 2, "-")]
            assert (warning in result.notes) is warned
            assert [n for n in result.notes
                    if "does not record what it was trained on" in n] == (
                        [warning] if warned else [])

    with subtests.test(msg="given a path, the file is checked before anything loads it"):
        not_a_checkpoint = tmp_path / "cyto2.CP_model"
        not_a_checkpoint.write_text("<html>404</html>", encoding="utf-8")
        segmenter = _Segmenter()
        with pytest.raises(mz.ModelUnreadable, match="cyto2.CP_model"):
            mz.benchmark(replace(named, path=str(not_a_checkpoint)),
                         images=[_a_field()], field_names=["f1"],
                         segment_fn=segmenter, qc=False)
        assert segmenter.calls == []


# ---------------------------------------------------------------------------
# format_zoo and _human_bytes
# ---------------------------------------------------------------------------

def test_format_zoo_prints_each_footer_only_when_it_applies(subtests, tmp_path):
    """A clean listing is the table alone; each problem adds exactly its own footer."""
    path = tmp_path / "toxo.CP_model"
    digest = _checkpoint(path)
    clean = [
        mz.ModelEntry(key="toxo", name="toxo.CP_model", path=str(path), sha256=digest,
                      verified=True, trained_on="Toxoplasma plaque assay, 1120px crops",
                      trained_by="ada"),
        mz.ModelEntry(key="lab_nuc", name="lab_nuc.CP_model", source="remote",
                      uri="https://lab.example/nuc", sha256="b" * 64,
                      trained_on="HeLa nuclei, 20x", trained_by="grace"),
    ]
    mystery = mz.ModelEntry(key="mystery", name="mystery.CP_model", path=str(path),
                            notes=("does not start with a torch header",))
    unknown_footer = (
        "  1 model(s) do not record what they were trained on: mystery.CP_model. "
        "A Cellpose model fine-tuned on confluent 60x cells is not interchangeable "
        "with one trained on sparse 20x ones, and nothing here tells you which this is.")
    checksum_footer = ("  1 local model(s) have no checksum on record; run verify() "
                       "against a published digest before trusting one.")
    note_footer = "  ! mystery.CP_model: does not start with a torch header"
    cases = [
        # (case id, entries, expected trailing lines after the clean table)
        ("nothing wrong: the table and nothing else", clean, None),
        ("one model with all three problems", clean + [mystery],
         ["", unknown_footer, checksum_footer, "", note_footer]),
    ]
    for case_id, entries, footers in cases:
        with subtests.test(msg=case_id):
            lines = mz.format_zoo(entries).splitlines()
            assert lines[0] == f"Model zoo — {len(entries)} model(s)"
            assert any("toxo.CP_model" in line for line in lines[1:])
            assert any("lab_nuc.CP_model" in line for line in lines[1:])
            # Header, column names, rule, then one row per model.
            table = 3 + len(entries)
            assert all(line.startswith("  ") for line in lines[1:table])
            if footers is None:
                assert len(lines) == table
            else:
                assert len(lines) == table + len(footers)
                assert lines[-len(footers):] == footers
    with subtests.test(msg="an empty listing says how to fill it"):
        assert mz.format_zoo([]) == (
            "Model zoo: nothing found.\n  Scan a folder with discover_local(), or "
            "point $SPACR_MODEL_CATALOGUE at a catalogue file.")


def test_human_bytes_scales_to_the_largest_unit_and_unknown_for_nothing(subtests):
    """Bytes are whole, larger units one decimal, beyond GB stays in GB; nothing is 'unknown'."""
    cases = [
        ("None", None, mz.UNKNOWN),
        ("text", "lots", mz.UNKNOWN),
        ("zero", 0, mz.UNKNOWN),
        ("negative", -5, mz.UNKNOWN),
        ("one byte", 1, "1 B"),
        ("bytes", 512, "512 B"),
        ("just under a KB", 1023, "1023 B"),
        ("one KB", 1024, "1.0 KB"),
        ("kilobytes", 1536, "1.5 KB"),
        ("numeric string", "2048", "2.0 KB"),
        ("megabytes", 3 * 1024 ** 2, "3.0 MB"),
        ("gigabytes", int(2.5 * 1024 ** 3), "2.5 GB"),
        ("a terabyte stays in GB", 1024 ** 4, "1024.0 GB"),
        ("beyond a terabyte", 5 * 1024 ** 4, "5120.0 GB"),
    ]
    for case_id, size, expected in cases:
        with subtests.test(msg=case_id):
            assert mz._human_bytes(size) == expected


# ---------------------------------------------------------------------------
# Timepose release: _timepose_files, _load_timepose, fetch of a timepose entry
# ---------------------------------------------------------------------------

FILES = {
    "weights/w.pt": b"weights",
    "fakepose288/__init__.py": b"",
    "fakepose288/infer.py": (b"CONTRACT = 'spacr.timepose/1'\n"
                             b"def load_model(folder, device):\n"
                             b"    return {'folder': folder, 'device': device}\n"),
}
REVISION = "b" * 40


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record(files=FILES, contract="spacr.timepose/1", entry="fakepose288.infer"):
    return {"key": "timepose_288", "kind": "timepose", "name": "timepose_288",
            "repo_id": "someone/private-timepose", "revision": REVISION,
            "entry": entry, "contract": contract, "weights": "weights/w.pt",
            "display_name": "Timepose 288",
            "files": {name: _sha(data) for name, data in files.items()}}


def _writer(content=FILES, calls=None):
    """An ``hf_hub_download``-shaped double writing ``content[filename]``."""
    def download(repo_id, filename, revision, repo_type, local_dir, token):
        if calls is not None:
            calls.append((filename, token))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content[filename])
        return str(path)
    return download


def _forget_adapters():
    for name in list(sys.modules):
        if name.startswith("_spacr_timepose_288"):
            sys.modules.pop(name)


@pytest.fixture
def release(monkeypatch):
    """Register a fake Timepose release with an empty verification and model cache."""
    monkeypatch.setattr(mz, "BUNDLED_REMOTE_MODELS",
                        tuple(mz.BUNDLED_REMOTE_MODELS) + (_record(),))
    monkeypatch.setattr(mz, "_TIMEPOSE_VERIFIED", {})
    monkeypatch.setattr(mz, "_TIMEPOSE_MODELS", {})
    monkeypatch.setattr(mz, "_timepose_token", lambda token=None: token or "hf_x")
    _forget_adapters()
    yield _record()
    _forget_adapters()


class _Forbidden(Exception):
    """A Hugging Face error that carries an HTTP status."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


def test_timepose_files_downloads_once_and_runs_nothing_unverified(subtests, release,
                                                                   tmp_path, monkeypatch):
    """Missing files are fetched and verified; a cache hit fetches nothing; every failure is exact."""
    folder_of = lambda dest: dest / "timepose_288" / REVISION[:12]  # noqa: E731
    access_text = ("Your Hugging Face account cannot read someone/private-timepose, "
                   "the private repository of Timepose 288. Ask the spaCR maintainer "
                   "for access, or choose another timelapse_mode.")
    login_text = ("Timepose 288 is in a private Hugging Face repository "
                  "(someone/private-timepose). Sign in with an account that has been "
                  "given access: run `hf auth login` or set HF_TOKEN, then run again.")
    mismatch_text = ("weights/w.pt of Timepose 288 does not match its pinned sha256 "
                     "and was deleted; nothing from this download was run.")
    bad = dict(FILES, **{"weights/w.pt": b"other"})

    def raising(error):
        def download(**_kwargs):
            raise error
        return download

    def refuse_unlink(self, *args, **kwargs):
        raise OSError("locked")

    cases = [
        # (case id, downloader, token, hub module?, patch, exception, message,
        #  expected download calls, progress, weights left on disk)
        ("the hub downloader fetches every file with progress", None, None, True, None,
         None, None, sorted((n, "hf_x") for n in FILES), [(1, 3), (2, 3), (3, 3)],
         b"weights"),
        ("an injected downloader gets the given token", "writer", "given", False, None,
         None, None, sorted((n, "given") for n in FILES), [(1, 3), (2, 3), (3, 3)],
         b"weights"),
        ("an error that is not about access propagates",
         raising(RuntimeError("disk full")), None, False, None, RuntimeError,
         "disk full", [], [], None),
        ("a 403 is an access failure", raising(_Forbidden(403)), None, False, None,
         mz._TimeposeAccessError, access_text, [], [], None),
        ("a 500 is not", raising(_Forbidden(500)), None, False, None, _Forbidden,
         "HTTP 500", [], [], None),
        ("no token and no downloader asks for a login", None, None, False, "no token",
         mz._TimeposeAccessError, login_text, [], [], None),
        ("a wrong download is deleted and refused", "bad", None, False, None,
         mz.ChecksumMismatch, mismatch_text, None, [(1, 3), (2, 3)], None),
        ("a wrong download that cannot be deleted is still refused", "bad", None, False,
         "locked", mz.ChecksumMismatch, mismatch_text, None, [(1, 3), (2, 3)],
         b"other"),
    ]
    for index, (case_id, downloader, token, hub, patch, exc, message, calls, progress,
                left) in enumerate(cases):
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            dest = tmp_path / f"case{index}"
            seen, made = [], []
            if hub:
                fake = types.ModuleType("huggingface_hub")
                fake.hf_hub_download = _writer(calls=made)
                mp.setitem(sys.modules, "huggingface_hub", fake)
            if downloader == "writer":
                downloader = _writer(calls=made)
            elif downloader == "bad":
                downloader = _writer(bad, calls=made)
            if patch == "no token":
                mp.setattr(mz, "_timepose_token", lambda token=None: None)
            elif patch == "locked":
                mp.setattr(Path, "unlink", refuse_unlink)
            kwargs = dict(dest=dest, token=token, downloader=downloader,
                          progress=lambda done, total: seen.append((done, total)))
            if exc is None:
                assert mz._timepose_files("timepose_288", **kwargs) == folder_of(dest)
            else:
                with pytest.raises(exc) as caught:
                    mz._timepose_files("timepose_288", **kwargs)
                assert str(caught.value) == message
            if calls is not None:
                assert sorted(made) == calls
            assert seen == progress
            weights = folder_of(dest) / "weights" / "w.pt"
            assert (weights.read_bytes() if weights.exists() else None) == left

    with subtests.test(msg="a verified cache downloads nothing"):
        dest = tmp_path / "warm"
        mz._timepose_files("timepose_288", dest=dest, downloader=_writer())
        made = []
        assert mz._timepose_files("timepose_288", dest=dest,
                                  downloader=_writer(calls=made)) == folder_of(dest)
        assert made == []

    with subtests.test(msg="fetch of a timepose entry returns its weights file"), \
            monkeypatch.context() as mp:
        real = mz._timepose_files
        mp.setattr(mz, "_timepose_files",
                   lambda key, dest=None, **kw: real(key, dest=dest, downloader=_writer(),
                                                     **kw))
        entry = types.SimpleNamespace(kind="timepose", key="timepose_288")
        path = mz.fetch(entry, tmp_path / "fetched")
        assert path == folder_of(tmp_path / "fetched" / "timepose") / "weights" / "w.pt"
        assert path.read_bytes() == b"weights"


def test_load_timepose_imports_only_a_matching_adapter_once(subtests, release, tmp_path,
                                                            monkeypatch):
    """The adapter is imported after verification, must match the contract, and is cached."""
    alias = f"_spacr_timepose_288_{REVISION[:12]}"
    package_files = dict(FILES, **{"fakepose288/__init__.py": (
        b"CONTRACT = 'spacr.timepose/1'\n"
        b"def load_model(folder, device):\n    return 'm'\n")})
    broken_files = dict(FILES, **{"fakepose288/__init__.py": b"raise ImportError('boom')\n"})
    cases = [
        # (case id, record, files, exception, message, expected model or callable)
        ("a submodule adapter loads the model", _record(), FILES, None, None,
         lambda folder: {"folder": str(folder), "device": "cpu"}),
        ("a package-only entry is itself the adapter",
         _record(files=package_files, entry="fakepose288"), package_files, None, None,
         lambda folder: "m"),
        ("an adapter that fails to import is not left registered",
         _record(files=broken_files), broken_files, ImportError, "boom", None),
        ("an adapter with another contract is refused",
         _record(contract="spacr.timepose/9"), FILES, mz.ModelZooError,
         "Timepose 288 does not provide the interface this spaCR expects "
         "(spacr.timepose/9); update spaCR.", None),
    ]
    for index, (case_id, record, files, exc, message, model) in enumerate(cases):
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            _forget_adapters()
            mp.setattr(mz, "BUNDLED_REMOTE_MODELS", (record,))
            mp.setattr(mz, "_TIMEPOSE_MODELS", {})
            dest = tmp_path / f"case{index}"
            folder = dest / "timepose_288" / REVISION[:12]
            if exc is not None:
                with pytest.raises(exc) as caught:
                    mz._load_timepose("timepose_288", device="cpu", dest=dest,
                                      downloader=_writer(files))
                assert str(caught.value) == message
                assert mz._TIMEPOSE_MODELS == {}
                if exc is ImportError:
                    assert alias not in sys.modules
                continue
            adapter, loaded = mz._load_timepose("timepose_288", device="CPU", dest=dest,
                                                downloader=_writer(files))
            assert adapter.CONTRACT == "spacr.timepose/1"
            assert loaded == model(folder)
            assert mz._TIMEPOSE_MODELS == {
                ("timepose_288", REVISION, "cpu"): (adapter, loaded)}
            made = []
            again = mz._load_timepose("timepose_288", device="cpu", dest=dest,
                                      downloader=_writer(files, calls=made))
            assert again[0] is adapter and again[1] is loaded
            assert made == []


def test_on_the_qt_gui_thread_is_false_off_the_gui_thread_and_without_qt(subtests, monkeypatch):
    """Only Qt's own GUI thread answers True; a worker thread or a process without Qt answers False."""
    import threading

    with subtests.test(msg="a worker thread is never the GUI thread"):
        seen = []
        worker = threading.Thread(target=lambda: seen.append(mz._on_the_qt_gui_thread()))
        worker.start()
        worker.join()
        assert seen == [False]
    with subtests.test(msg="no PySide6 at all"):
        monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)
        assert mz._on_the_qt_gui_thread() is False
