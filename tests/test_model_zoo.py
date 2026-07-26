"""Tests for the Model Zoo — ``spacr.model_zoo``.

Every test here is **offline and CPU-only**, and that is a property under test
rather than a convention: :func:`test_no_network_is_attempted_on_the_offline_paths`
replaces ``socket.socket`` with a landmine and then runs discovery, the
catalogue, checksums, ranking and reporting over it. Downloads are exercised
for real, through ``file://`` URIs and injected openers, so the atomicity,
checksum and versioning paths get run rather than mocked past.

The properties pinned here are the ones that decide whether the zoo is
trustworthy rather than merely convenient:

* it finds the checkpoints on a machine and tells Cellpose models from
  classifiers — and ignores the CSVs, PNGs and ``.npy`` masks around them;
* a checksum passes on the right bytes and **fails on one changed byte**;
* an interrupted or cancelled download leaves **nothing** at the destination;
* a second download of the same name lands as a new version, never on top;
* missing provenance reads ``'unknown'``, never blank;
* benchmarks from **different field sets are refused, not sorted**;
* a corrupt or wrong-architecture checkpoint fails with the filename in the
  message, not a torch ``KeyError``;
* ``benchmark`` delegates to :mod:`spacr.model_compare` rather than
  reimplementing segmentation;
* importing the module pulls in neither torch nor cellpose.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace

import numpy as np
import pytest

from spacr import model_zoo as zoo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: A minimal file that passes the torch magic-byte check without being torch.
TORCH_ZIP_HEADER = b"PK\x03\x04"


def write_checkpoint(path, payload: bytes = b"weights") -> str:
    """Write a file that looks like a torch checkpoint. Returns its digest."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(TORCH_ZIP_HEADER + payload)
    return hashlib.sha256(TORCH_ZIP_HEADER + payload).hexdigest()


def write_settings(path, mapping) -> None:
    """Write a ``Key,Value`` settings CSV the way ``save_settings`` does."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("Key,Value\n")
        for key, value in mapping.items():
            fh.write(f"{key},{value}\n")


def a_field(size: int = 40, seed: int = 0) -> np.ndarray:
    """A deterministic field with two bright blobs."""
    image = np.full((size, size), 100.0, dtype=np.float32)
    image[4:14, 4:14] = 800.0 + seed
    image[24:34, 24:34] = 600.0 + seed
    return image


def masks_two_objects(size: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.int32)
    mask[4:14, 4:14] = 1
    mask[24:34, 24:34] = 2
    return mask


class FakeSegmenter:
    """Stands in for Cellpose and records exactly what it was asked to run."""

    def __init__(self, mask=None):
        self.mask = masks_two_objects() if mask is None else mask
        self.calls = []

    def __call__(self, images, config):
        self.calls.append((config.name, config.resolved_model, len(images),
                           config.eval_kwargs()))
        return [self.mask.copy() for _ in images]


def an_entry(path="", **kwargs) -> zoo.ModelEntry:
    """A ModelEntry with sane defaults for the tests that do not care."""
    fields = dict(key="k", name=os.path.basename(str(path)) or "model.CP_model",
                  kind="cellpose", source="local", path=str(path))
    fields.update(kwargs)
    return zoo.ModelEntry(**fields)


@pytest.fixture
def zoo_tree(tmp_path):
    """A folder shaped the way spaCR actually leaves models on disk.

    * a Cellpose model with the bundled pack's ``<file>_settings.csv`` sidecar;
    * a Cellpose model with no provenance at all;
    * a classifier checkpoint inside a training ``dst`` with the per-run
      ``settings.csv`` :func:`spacr.deep_spacr.train_test_model` writes;
    * and a pile of things that are not models.
    """
    root = tmp_path / "screen1"
    cellpose = root / "models" / "cellpose_model"
    named = cellpose / "toxo_cyto_e25_X512_Y512.CP_model"
    write_checkpoint(named)
    write_settings(cellpose / "toxo_cyto_e25_X512_Y512.CP_model_settings.csv", {
        "img_src": "/data/plaques/train",
        "model_name": "toxo",
        "n_epochs": 25,
        "diameter": 30,
        "width_height": "[512, 512]",
        "grayscale": True,
    })
    write_checkpoint(cellpose / "mystery.CP_model")

    run = root / "model" / "maxvit_t" / "rgb" / "epochs_10"
    write_checkpoint(run / "maxvit_t_epoch_10_channels_rgb.pth")
    write_settings(run / "settings.csv", {
        "src": "/data/screen1",
        "model_type": "maxvit_t",
        "image_size": 224,
        "epochs": 10,
        "classes": "['nc', 'pc']",
    })
    (run / "train.csv").write_text(
        "epoch,accuracy,loss\n1,0.60,0.9\n2,0.80,0.4\n")
    (run / "validation.csv").write_text(
        "epoch,accuracy,loss\n1,0.55,1.0\n2,0.75,0.5\n")

    # Not models. None of these may appear in a listing.
    (root / "notes.txt").write_text("hello")
    (root / "montage.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    np.save(root / "cell_mask.npy", masks_two_objects())
    write_settings(root / "settings" / "unrelated.csv", {"a": 1})
    (cellpose / "README").write_text("read me")
    return root


# ---------------------------------------------------------------------------
# 1. the module imports nothing heavy
# ---------------------------------------------------------------------------

def test_importing_the_module_does_not_import_torch_or_cellpose():
    """Browsing the zoo has to work on a machine with neither installed.

    Checked in a fresh interpreter, because another test in the session may
    already have imported torch for its own reasons.

    ``PYTHONPATH`` is rebuilt from scratch rather than inherited: the coverage
    runner puts a ``sitecustomize.py`` on it that pre-imports torch (to dodge a
    ``_has_torch_function`` crash under tracing), and inheriting that would
    make this assertion pass or fail depending on how the suite was launched.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys; import spacr.model_zoo as z; "
        "z.catalogue(); z.format_zoo([]); "
        "print('torch' in sys.modules, 'cellpose' in sys.modules)"
    )
    env = dict(os.environ, PYTHONPATH=root)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=root, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False False"), out.stdout


def test_no_network_is_attempted_on_the_offline_paths(monkeypatch, zoo_tree,
                                                      tmp_path):
    """Every browse/verify/report path must be provably local.

    ``socket.socket`` is replaced with something that raises, so *any*
    connection attempt — requests, huggingface_hub, urllib — fails the test
    rather than quietly succeeding on a machine with a network.
    """
    import socket

    def _boom(*_a, **_k):
        raise AssertionError("a network connection was attempted")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom, raising=False)

    entries = zoo.discover_local(zoo_tree) + zoo.catalogue()
    assert entries
    zoo.format_zoo(entries)
    for entry in entries:
        entry.describe()
        entry.summary_line()
        if entry.exists:
            zoo.sha256_file(entry.path)
            zoo.inspect_checkpoint(entry.path)
    zoo.resolve(entries[0].key, entries)
    zoo.versioned_path(tmp_path, "x.CP_model")
    zoo.rank_groups([])
    zoo.format_benchmarks([])


# ---------------------------------------------------------------------------
# 2. discovery
# ---------------------------------------------------------------------------

def test_discovery_finds_the_checkpoints_and_classifies_them(zoo_tree):
    entries = zoo.discover_local(zoo_tree)
    by_name = {e.name: e for e in entries}
    assert set(by_name) == {
        "toxo_cyto_e25_X512_Y512.CP_model",
        "mystery.CP_model",
        "maxvit_t_epoch_10_channels_rgb.pth",
    }
    assert by_name["toxo_cyto_e25_X512_Y512.CP_model"].kind == "cellpose"
    assert by_name["mystery.CP_model"].kind == "cellpose"
    assert by_name["maxvit_t_epoch_10_channels_rgb.pth"].kind == "classifier"
    # Cellpose first, then by name — a stable order for a list widget.
    assert [e.kind for e in entries] == ["cellpose", "cellpose", "classifier"]


def test_discovery_ignores_everything_that_is_not_a_model(zoo_tree):
    names = {e.name for e in zoo.discover_local(zoo_tree)}
    for junk in ("notes.txt", "montage.png", "cell_mask.npy",
                 "unrelated.csv", "README",
                 "toxo_cyto_e25_X512_Y512.CP_model_settings.csv"):
        assert junk not in names, f"{junk} was listed as a model"


def test_a_pth_inside_a_cellpose_folder_is_a_cellpose_model(tmp_path):
    """``.pth`` is the classifier suffix, but not inside a Cellpose folder.

    ``cellpose.train.train_seg`` writes into ``<save_path>/models/``, and a
    checkpoint mislabelled as a classifier would be handed to the wrong loader.
    """
    write_checkpoint(tmp_path / "models" / "cellpose_model" / "custom.pth")
    write_checkpoint(tmp_path / "runs" / "epochs_5" / "resnet50_epoch_5.pth")
    kinds = {e.name: e.kind for e in zoo.discover_local(tmp_path)}
    assert kinds["custom.pth"] == "cellpose"
    assert kinds["resnet50_epoch_5.pth"] == "classifier"


def test_an_extensionless_file_is_a_model_only_if_it_looks_like_one(tmp_path):
    """A README in a Cellpose folder must not become a model.

    ``cellpose.train`` saves ``<save_path>/models/<name>`` with no suffix, so
    the folder is a candidate; the four magic bytes are what separate the
    checkpoint from the documentation next to it.
    """
    models = tmp_path / "models"
    write_checkpoint(models / "cpsam")
    (models / "README").write_text("not a model")
    (models / "train.log").write_text("epoch 1")
    names = {e.name for e in zoo.discover_local(tmp_path)}
    assert names == {"cpsam"}


def test_discovery_never_hashes_unless_asked(zoo_tree):
    """Hashing every checkpoint to fill a list widget is minutes, not milliseconds."""
    assert all(not e.sha256 for e in zoo.discover_local(zoo_tree))
    hashed = zoo.discover_local(zoo_tree, compute_hashes=True)
    assert all(len(e.sha256) == 64 for e in hashed)
    assert all(e.checksum_state == "recorded" for e in hashed)


# ---------------------------------------------------------------------------
# 3. provenance
# ---------------------------------------------------------------------------

def test_provenance_comes_off_the_settings_snapshot_spacr_already_writes(zoo_tree):
    entry = next(e for e in zoo.discover_local(zoo_tree)
                 if e.name.startswith("toxo_cyto"))
    assert "/data/plaques/train" in entry.trained_on
    assert "diameter 30" in entry.trained_on
    assert "25 epochs" in entry.trained_on
    assert entry.settings_path.endswith("_settings.csv")
    assert entry.provenance_known


def test_classifier_provenance_is_read_through_train_compare(zoo_tree):
    """The classifier half reuses ``train_compare.load_run`` rather than
    growing a second discovery pass over the same folders."""
    entry = next(e for e in zoo.discover_local(zoo_tree)
                 if e.kind == "classifier")
    assert "/data/screen1" in entry.trained_on
    assert "maxvit_t" in entry.trained_on
    assert entry.settings_path.endswith("settings.csv")
    # Both the best and the last validation epoch, never one alone.
    text = " ".join(entry.metrics)
    assert "best accuracy" in text and "last accuracy" in text


def test_missing_provenance_says_unknown_and_is_never_blank(zoo_tree):
    """A blank cell reads as 'no constraints'; that is the opposite of the truth."""
    entry = next(e for e in zoo.discover_local(zoo_tree)
                 if e.name == "mystery.CP_model")
    assert entry.trained_on == zoo.UNKNOWN == "unknown"
    assert entry.trained_by == zoo.UNKNOWN
    assert entry.trained_on != ""
    assert not entry.provenance_known
    assert any("unknown" in n for n in entry.notes)
    # …and it says so in the listing, not only in the object.
    assert "unknown" in zoo.format_zoo([entry])
    assert "do not record what they were trained on" in zoo.format_zoo([entry])


def test_blank_provenance_passed_in_is_normalised_to_unknown():
    entry = zoo.ModelEntry(key="k", name="n", trained_on="", trained_by="   ")
    assert entry.trained_on == "unknown"
    assert entry.trained_by == "unknown"


def test_a_bad_kind_is_refused_at_construction():
    with pytest.raises(ValueError, match="kind must be one of"):
        zoo.ModelEntry(key="k", name="n", kind="diffusion")


# ---------------------------------------------------------------------------
# 4. checksums
# ---------------------------------------------------------------------------

def test_verification_passes_on_the_right_bytes(tmp_path):
    digest = write_checkpoint(tmp_path / "m.CP_model", b"a" * 1000)
    entry = an_entry(tmp_path / "m.CP_model", sha256=digest)
    assert zoo.verify(entry) is True


def test_verification_fails_when_one_byte_changed(tmp_path):
    """The whole point: a swapped or truncated checkpoint still *loads*.

    It does not raise, it does not warn — it produces different masks, and the
    run that used it is indistinguishable from the run that did not.
    """
    path = tmp_path / "m.CP_model"
    digest = write_checkpoint(path, b"a" * 1000)
    entry = an_entry(path, sha256=digest)
    assert zoo.verify(entry) is True

    data = bytearray(path.read_bytes())
    data[500] ^= 0x01                       # exactly one bit of one byte
    path.write_bytes(bytes(data))
    assert len(path.read_bytes()) == len(data)   # same size, different bytes
    assert zoo.verify(entry) is False


def test_verifying_without_a_digest_is_an_error_not_a_false(tmp_path):
    """"Nobody said what right looks like" is not the same as "this is wrong"."""
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    with pytest.raises(zoo.ModelZooError, match="no checksum recorded"):
        zoo.verify(an_entry(path))


def test_verifying_a_missing_file_names_the_file(tmp_path):
    entry = an_entry(tmp_path / "gone.CP_model", sha256="0" * 64)
    with pytest.raises(zoo.ModelUnreadable, match="gone.CP_model"):
        zoo.verify(entry)


def test_verifying_a_model_that_was_never_downloaded_says_so():
    entry = zoo.ModelEntry(key="k", name="remote.CP_model", source="remote",
                           uri="https://example/remote.CP_model",
                           sha256="0" * 64)
    with pytest.raises(zoo.ModelUnreadable, match="has not been downloaded"):
        zoo.verify(entry)


# ---------------------------------------------------------------------------
# 5. fetching — atomic, checksummed, versioned
# ---------------------------------------------------------------------------

@pytest.fixture
def remote(tmp_path):
    """A 'remote' model served from a local file, so fetch runs for real."""
    source = tmp_path / "server" / "hela_60x.CP_model"
    digest = write_checkpoint(source, b"h" * 4096)
    entry = zoo.ModelEntry(
        key="hela_60x", name="hela_60x.CP_model", source="remote",
        uri=f"file://{source}", sha256=digest, size_bytes=source.stat().st_size,
        trained_on="HeLa, 60x, confluent", trained_by="A. Researcher")
    return entry, source, digest


def test_a_good_download_lands_verified_with_the_hash_it_computed(remote, tmp_path):
    entry, source, digest = remote
    dest = tmp_path / "dest"
    installed = zoo.install(entry, dest)
    assert installed.path == str((dest / "hela_60x.CP_model").resolve())
    assert installed.sha256 == digest
    assert installed.verified is True
    assert installed.checksum_state == "verified"
    assert installed.source == "local"
    # Provenance survives the trip — it is the reason for having a catalogue.
    assert installed.trained_on == "HeLa, 60x, confluent"
    assert (dest / "hela_60x.CP_model").read_bytes() == source.read_bytes()


def test_a_checksum_mismatch_installs_nothing(remote, tmp_path):
    entry, _source, digest = remote
    dest = tmp_path / "dest"
    dest.mkdir()
    wrong = replace(entry, sha256="f" * 64)
    with pytest.raises(zoo.ChecksumMismatch) as excinfo:
        zoo.fetch(wrong, dest)
    message = str(excinfo.value)
    assert "hela_60x.CP_model" in message
    assert "f" * 64 in message and digest in message
    assert list(dest.iterdir()) == [], "a mismatched download was left behind"


def test_an_interrupted_fetch_leaves_nothing_at_the_destination(remote, tmp_path):
    """The atomicity test.

    A download that dies halfway must not leave a file that looks like a model:
    a truncated checkpoint loads, segments, and is wrong. Not even the
    temporary file may survive.
    """
    entry, _source, _digest = remote
    dest = tmp_path / "dest"

    def dying_opener(uri):
        def chunks():
            yield b"PK\x03\x04" + b"h" * 100
            raise ConnectionError("the network went away")
        return chunks(), 4100

    with pytest.raises(ConnectionError):
        zoo.fetch(entry, dest, opener=dying_opener)
    assert list(dest.iterdir()) == []


def test_a_cancelled_fetch_leaves_nothing_at_the_destination(remote, tmp_path):
    entry, _source, _digest = remote
    dest = tmp_path / "dest"
    stop = {"now": False}

    def slow_opener(uri):
        def chunks():
            for _ in range(50):
                stop["now"] = True          # cancel arrives after chunk 1
                yield b"h" * 64
        return chunks(), 3200

    with pytest.raises(zoo.DownloadCancelled, match="hela_60x.CP_model"):
        zoo.fetch(entry, dest, opener=slow_opener,
                  cancel=lambda: stop["now"])
    assert list(dest.iterdir()) == []


def test_an_empty_response_installs_nothing(remote, tmp_path):
    entry, _source, _digest = remote
    dest = tmp_path / "dest"
    with pytest.raises(zoo.ModelZooError, match="returned no data"):
        zoo.fetch(entry, dest, opener=lambda uri: (iter(()), 0))
    assert list(dest.iterdir()) == []


def test_fetching_twice_versions_the_destination_and_never_overwrites(remote,
                                                                     tmp_path):
    """Two models with the same filename are normal; losing one is not."""
    entry, source, _digest = remote
    dest = tmp_path / "dest"
    first = zoo.fetch(entry, dest)
    assert first.name == "hela_60x.CP_model"
    original = first.read_bytes()

    # The author retrained and republished under the same name.
    new_digest = write_checkpoint(source, b"retrained" * 100)
    second_entry = replace(entry, sha256=new_digest)
    second = zoo.fetch(second_entry, dest)

    assert second.name == "hela_60x_v2.CP_model"
    assert second != first
    assert first.read_bytes() == original, "the first model was overwritten"
    assert second.read_bytes() != original

    third = zoo.fetch(second_entry, dest)
    assert third.name == "hela_60x_v3.CP_model"
    assert zoo.install(second_entry, dest).version == "4"


def test_versioned_path_counts_on_from_an_existing_version(tmp_path):
    (tmp_path / "m_v2.CP_model").write_bytes(b"x")
    assert zoo.versioned_path(tmp_path, "m_v2.CP_model").name == "m_v3.CP_model"
    assert zoo.versioned_path(tmp_path, "m.CP_model").name == "m.CP_model"


def test_an_unverifiable_download_is_refused_by_default(tmp_path):
    """A download nobody can check is exactly what this module exists to stop
    being routine — so it takes an explicit decision, not a default."""
    source = tmp_path / "server" / "nohash.CP_model"
    write_checkpoint(source)
    entry = zoo.ModelEntry(key="nohash", name="nohash.CP_model",
                           source="remote", uri=f"file://{source}")
    dest = tmp_path / "dest"
    with pytest.raises(zoo.ModelZooError, match="no sha256 was published"):
        zoo.fetch(entry, dest)
    assert not dest.exists() or list(dest.iterdir()) == []

    installed = zoo.install(entry, dest, require_checksum=False)
    assert installed.verified is False
    assert installed.checksum_state == "recorded"
    assert len(installed.sha256) == 64
    assert any("proves nothing about where they came from" in n
               for n in installed.notes)


def test_fetch_reports_progress_and_the_total_size(remote, tmp_path):
    entry, _source, _digest = remote
    seen = []
    zoo.fetch(entry, tmp_path / "dest", chunk_size=512,
              progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (0, entry.size_bytes)
    assert seen[-1] == (entry.size_bytes, entry.size_bytes)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_fetch_refuses_a_scheme_it_does_not_speak(tmp_path):
    entry = zoo.ModelEntry(key="k", name="m.CP_model", source="remote",
                           uri="ftp://example.org/m.CP_model", sha256="a" * 64)
    with pytest.raises(zoo.ModelZooError, match="do not know how to fetch"):
        zoo.fetch(entry, tmp_path / "dest")


def test_fetch_needs_a_uri():
    with pytest.raises(zoo.ModelZooError, match="no uri"):
        zoo.fetch(an_entry("/nowhere/m.CP_model", sha256="a" * 64), "/tmp")


# ---------------------------------------------------------------------------
# 6. unreadable / wrong checkpoints
# ---------------------------------------------------------------------------

def test_a_file_that_is_not_a_checkpoint_names_the_file(tmp_path):
    """The default failure is a torch KeyError on a state-dict key, which names
    nothing the user chose and reads like a spaCR bug."""
    path = tmp_path / "not_a_model.CP_model"
    path.write_text("<html>404 Not Found</html>")
    with pytest.raises(zoo.ModelUnreadable) as excinfo:
        zoo.inspect_checkpoint(path)
    message = str(excinfo.value)
    assert "not_a_model.CP_model" in message
    assert "not a PyTorch checkpoint" in message
    assert "KeyError" not in message


def test_an_empty_checkpoint_names_the_file(tmp_path):
    path = tmp_path / "truncated.pth"
    path.write_bytes(b"")
    with pytest.raises(zoo.ModelUnreadable, match=r"truncated\.pth is empty"):
        zoo.inspect_checkpoint(path)


def test_a_missing_checkpoint_names_the_file(tmp_path):
    with pytest.raises(zoo.ModelUnreadable, match="no such model file"):
        zoo.inspect_checkpoint(tmp_path / "gone.pth")


def test_a_wrong_architecture_checkpoint_names_the_file_not_a_state_dict_key(
        tmp_path):
    """A Cellpose checkpoint handed to the classifier loader raises
    ``KeyError('state_dict')`` from three frames inside torch. That gets
    translated, with the filename in it."""
    path = tmp_path / "wrong_arch.pth"
    write_checkpoint(path)

    def loader(_p):
        raise KeyError("state_dict")

    with pytest.raises(zoo.ModelUnreadable) as excinfo:
        zoo.inspect_checkpoint(path, loader=loader, deep=True)
    message = str(excinfo.value)
    assert "wrong_arch.pth" in message
    assert "could not be loaded as a model checkpoint" in message
    assert "KeyError" in message      # the cause is named, not the whole error
    assert not isinstance(excinfo.value, KeyError)


def test_a_shallow_inspection_reports_the_format_without_loading(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path, b"x" * 32)
    info = zoo.inspect_checkpoint(path)
    assert info["format"] == "zip"
    assert info["loaded"] is False
    assert info["size_bytes"] == 36


def test_a_git_lfs_pointer_is_listed_but_flagged(tmp_path):
    """The bundled models are LFS-tracked; a checkout without LFS leaves a
    120-byte text pointer where the weights should be."""
    path = tmp_path / "models" / "cellpose_model" / "lfs.CP_model"
    os.makedirs(path.parent, exist_ok=True)
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:…\n")
    entry = zoo.entry_from_file(path)
    assert any("Git LFS pointer" in n for n in entry.notes)
    with pytest.raises(zoo.ModelUnreadable, match="lfs.CP_model"):
        zoo.inspect_checkpoint(path)


# ---------------------------------------------------------------------------
# 7. catalogue and resolution
# ---------------------------------------------------------------------------

def test_the_bundled_catalogue_declares_its_provenance_and_its_missing_hash():
    entry = zoo._entry_from_mapping(zoo.BUNDLED_REMOTE_MODELS[0])
    assert entry.uri.startswith(
        f"https://huggingface.co/datasets/{zoo.HF_MODELS_REPO}/resolve/main/")
    assert entry.trained_on != zoo.UNKNOWN
    assert entry.trained_by != zoo.UNKNOWN
    assert entry.checksum_state == "none"
    assert any("no published checksum" in n for n in entry.notes)


def test_a_catalogue_file_is_read_and_its_hashes_are_what_make_it_useful(tmp_path):
    path = tmp_path / "zoo.json"
    path.write_text(json.dumps({"models": [{
        "key": "hela_60x",
        "name": "hela_60x.CP_model",
        "kind": "cellpose",
        "uri": "https://example.org/hela_60x.CP_model",
        "sha256": "AB" * 32,
        "size_bytes": 10,
        "trained_on": "HeLa, 60x, confluent monolayer",
        "trained_by": "A. Researcher, 2026-02",
    }]}))
    entry, = zoo.load_catalogue_file(path)
    assert entry.sha256 == "ab" * 32           # normalised to lowercase
    # A promise about bytes that are not here yet — not "verified", and not
    # "none" either.
    assert entry.checksum_state == "published"
    assert entry.trained_on == "HeLa, 60x, confluent monolayer"


def test_a_broken_catalogue_file_names_itself(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(zoo.ModelZooError, match="bad.json"):
        zoo.load_catalogue_file(path)
    with pytest.raises(zoo.ModelZooError, match="no such catalogue file"):
        zoo.load_catalogue_file(tmp_path / "absent.json")


def test_the_catalogue_picks_up_a_configured_catalogue_file(tmp_path,
                                                            monkeypatch):
    path = tmp_path / "zoo.json"
    path.write_text(json.dumps([{"name": "extra.CP_model", "sha256": "c" * 64,
                                 "trained_on": "u2os 20x"}]))
    monkeypatch.setenv(zoo.CATALOGUE_ENV_VAR, str(path))
    names = {e.name for e in zoo.catalogue()}
    assert "extra.CP_model" in names


def test_resolve_prefers_a_path_that_exists(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    assert zoo.resolve(str(path)).path == str(path.resolve())


def test_resolve_finds_an_entry_by_key_or_name(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    entries = zoo.discover_local(tmp_path)
    assert zoo.resolve("m.CP_model", entries).name == "m.CP_model"
    assert zoo.resolve(entries[0].key, entries).key == entries[0].key


def test_resolve_says_which_it_could_not_find(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    entries = zoo.discover_local(tmp_path)
    with pytest.raises(zoo.ModelZooError, match="no model called 'nope'"):
        zoo.resolve("nope", entries)
    with pytest.raises(zoo.ModelUnreadable, match="no such model file"):
        zoo.resolve(str(tmp_path / "sub" / "gone.CP_model"), entries)


# ---------------------------------------------------------------------------
# 8. benchmarking — delegation, not reimplementation
# ---------------------------------------------------------------------------

def test_benchmark_delegates_segmentation_to_model_compare(tmp_path,
                                                           monkeypatch):
    """The default backend is ``model_compare.segment_with_cellpose``, called
    with a ``model_compare.ModelConfig`` — no second Cellpose call site."""
    import spacr.model_compare as mc

    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    entry = zoo.entry_from_file(path)
    calls = []

    def fake_segment(images, config):
        calls.append((images, config))
        return [masks_two_objects() for _ in images]

    monkeypatch.setattr(mc, "segment_with_cellpose", fake_segment)
    result = zoo.benchmark(entry, images=[a_field(), a_field(seed=1)],
                           field_names=["f1", "f2"])

    assert len(calls) == 1
    images, config = calls[0]
    assert len(images) == 2
    assert isinstance(config, mc.ModelConfig)
    # The checkpoint is passed by path, which is what _choose_model loads as
    # pretrained_model — the bug that made custom models silently unusable.
    assert config.resolved_model == str(path.resolve())
    assert result.total_objects == 4
    assert result.fields == ["f1", "f2"]


def test_benchmark_surfaces_the_arguments_cellpose_4_ignores(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    entry = zoo.entry_from_file(path)
    result = zoo.benchmark(entry, images=[a_field()], field_names=["f1"],
                           segment_fn=FakeSegmenter(),
                           settings={"diameter": 45.0, "diam_mean": 17})
    assert result.honoured["diameter"] == 45.0
    assert result.ignored["diam_mean"] == 17
    assert any("diam_mean" in n and "ignored" in n for n in result.notes)


def test_benchmark_loads_fields_through_model_compare(tmp_path):
    folder = tmp_path / "plate1" / "1"
    folder.mkdir(parents=True)
    for i in range(4):
        np.save(folder / f"field_{i}.npy", a_field(seed=i))
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)

    result = zoo.benchmark(zoo.entry_from_file(path), source=folder,
                           n_fields=3, segment_fn=FakeSegmenter())
    assert result.n_fields == 3
    assert str(folder) in result.fieldset_label


def test_benchmark_scores_every_field_with_seg_qc(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    empty = np.zeros((40, 40), dtype=np.int32)
    result = zoo.benchmark(
        zoo.entry_from_file(path),
        images=[a_field(), a_field(seed=1)], field_names=["f1", "f2"],
        segment_fn=lambda images, config: [masks_two_objects(), empty])
    assert [r.n_objects for r in result.rows] == [2, 0]
    assert result.rows[1].severity in ("warn", "fail")
    assert result.rows[1].flags
    assert result.n_ok <= 1


def test_benchmark_refuses_a_checkpoint_that_is_not_one(tmp_path):
    path = tmp_path / "m.CP_model"
    path.write_text("not a checkpoint")
    entry = zoo.entry_from_file(path)

    def never(*_a, **_k):
        raise AssertionError("segmentation must not start on a bad checkpoint")

    with pytest.raises(zoo.ModelUnreadable, match="m.CP_model"):
        zoo.benchmark(entry, images=[a_field()], segment_fn=never)


def test_benchmark_says_when_the_model_has_no_provenance(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    result = zoo.benchmark(zoo.entry_from_file(path), images=[a_field()],
                           segment_fn=FakeSegmenter())
    assert any("does not record what it was trained on" in n
               for n in result.notes)


def test_benchmark_refuses_a_wrong_number_of_masks(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    with pytest.raises(ValueError, match="returned 1 mask"):
        zoo.benchmark(zoo.entry_from_file(path),
                      images=[a_field(), a_field(seed=1)],
                      segment_fn=lambda images, config: [masks_two_objects()])


def test_benchmark_needs_something_to_run_on(tmp_path):
    path = tmp_path / "m.CP_model"
    write_checkpoint(path)
    entry = zoo.entry_from_file(path)
    with pytest.raises(ValueError, match="either images= or source="):
        zoo.benchmark(entry)
    with pytest.raises(ValueError, match="at least one image"):
        zoo.benchmark(entry, images=[])


def test_compare_entries_hands_both_models_to_compare_models(tmp_path,
                                                             monkeypatch):
    """Two models head to head is ``model_compare.compare_models``, unchanged —
    the zoo does not grow a second, weaker comparison."""
    import spacr.model_compare as mc

    a, b = tmp_path / "a.CP_model", tmp_path / "b.CP_model"
    write_checkpoint(a)
    write_checkpoint(b, b"other")
    seen = {}

    def fake_compare(images, config_a, config_b, **kwargs):
        seen.update(images=images, a=config_a, b=config_b, kwargs=kwargs)
        return "report"

    monkeypatch.setattr(mc, "compare_models", fake_compare)
    out = zoo.compare_entries(zoo.entry_from_file(a), zoo.entry_from_file(b),
                              images=[a_field()], field_names=["f1"])
    assert out == "report"
    assert seen["a"].resolved_model == str(a.resolve())
    assert seen["b"].resolved_model == str(b.resolve())
    assert seen["kwargs"]["field_names"] == ["f1"]


# ---------------------------------------------------------------------------
# 9. the refusal: incomparable benchmarks
# ---------------------------------------------------------------------------

def a_result(name: str, names, images, qc="ok", seconds=1.0):
    """A BenchmarkResult whose fieldset id comes from the actual pixels."""
    entry = zoo.ModelEntry(key=name, name=name, trained_on="synthetic")
    return zoo.BenchmarkResult(
        entry=entry,
        fieldset=zoo.fieldset_id(names, images),
        fieldset_label=zoo._fieldset_label(names, None),
        rows=[zoo.FieldBenchmark(field=n, n_objects=5, severity=qc)
              for n in names],
        seconds=seconds,
    )


def test_the_fieldset_id_is_the_pixels_not_the_folder_name():
    names = ["f1", "f2"]
    first = [a_field(), a_field(seed=1)]
    same_pixels_elsewhere = [a_field(), a_field(seed=1)]
    other = [a_field(seed=7), a_field(seed=8)]

    assert zoo.fieldset_id(names, first) == \
        zoo.fieldset_id(names, same_pixels_elsewhere)
    assert zoo.fieldset_id(names, first) != zoo.fieldset_id(names, other)
    # Same pixels, different fields chosen out of the plate: not the same set.
    assert zoo.fieldset_id(["f1", "f3"], first) != zoo.fieldset_id(names, first)


def test_benchmarks_on_different_field_sets_are_refused_not_sorted():
    """The failure mode this module exists to prevent.

    Two models benchmarked on different images produce two numbers, and two
    numbers always sort. The resulting ranking looks exactly like a real one.
    """
    plate1 = (["f1", "f2"], [a_field(), a_field(seed=1)])
    plate2 = (["g1", "g2"], [a_field(seed=7), a_field(seed=8)])
    good_on_plate1 = a_result("A.CP_model", *plate1, qc="ok")
    bad_on_plate2 = a_result("B.CP_model", *plate2, qc="fail")

    assert good_on_plate1.fieldset != bad_on_plate2.fieldset
    with pytest.raises(zoo.IncomparableBenchmarks) as excinfo:
        zoo.rank([good_on_plate1, bad_on_plate2])
    message = str(excinfo.value)
    assert "2 different field sets" in message
    assert "A.CP_model" in message and "B.CP_model" in message
    assert "rank_groups()" in message


def test_ranking_inside_one_field_set_is_allowed_and_ordered():
    names, images = ["f1", "f2"], [a_field(), a_field(seed=1)]
    broken = a_result("broken.CP_model", names, images, qc="fail", seconds=1.0)
    fine = a_result("fine.CP_model", names, images, qc="ok", seconds=9.0)
    quick = a_result("quick.CP_model", names, images, qc="ok", seconds=0.5)

    assert [r.entry.name for r in zoo.rank([broken, fine, quick])] == \
        ["quick.CP_model", "fine.CP_model", "broken.CP_model"]
    assert [r.entry.name for r in zoo.rank([broken, fine, quick],
                                           key="seconds")] == \
        ["quick.CP_model", "broken.CP_model", "fine.CP_model"]


def test_rank_groups_keeps_the_field_sets_apart_instead_of_refusing():
    plate1 = (["f1"], [a_field()])
    plate2 = (["g1"], [a_field(seed=7)])
    results = [a_result("A", *plate1), a_result("B", *plate2),
               a_result("C", *plate1, qc="fail")]
    groups = zoo.rank_groups(results)
    assert len(groups) == 2
    ranked = groups[results[0].fieldset]
    assert [r.entry.name for r in ranked] == ["A", "C"]


def test_the_report_groups_incomparable_benchmarks_and_says_so():
    plate1 = (["f1"], [a_field()])
    plate2 = (["g1"], [a_field(seed=7)])
    text = zoo.format_benchmarks([a_result("A", *plate1),
                                  a_result("B", *plate2)])
    assert "2 field set(s)" in text
    assert "DIFFERENT fields and are not comparable" in text
    assert text.count("field set ") >= 2
    assert "not an accuracy" in text


def test_a_single_field_set_report_does_not_shout_about_comparability():
    names, images = ["f1"], [a_field()]
    text = zoo.format_benchmarks([a_result("A", names, images),
                                  a_result("B", names, images)])
    assert "DIFFERENT fields" not in text
    assert "1 field set(s)" in text


def test_there_is_no_accuracy_rank_key():
    """A key called 'score' would be inventing a ground truth the zoo has not got."""
    assert set(zoo.RANK_KEYS) == {"qc", "seconds"}
    names, images = ["f1"], [a_field()]
    with pytest.raises(ValueError, match="no ground truth"):
        zoo.rank([a_result("A", names, images)], key="accuracy")


def test_an_unscored_benchmark_does_not_rank_first():
    names, images = ["f1"], [a_field()]
    scored = a_result("scored", names, images, qc="ok")
    unscored = a_result("unscored", names, images, qc="-")
    assert unscored.qc_score != unscored.qc_score          # nan
    assert [r.entry.name for r in zoo.rank([unscored, scored])] == \
        ["scored", "unscored"]


# ---------------------------------------------------------------------------
# 10. reporting
# ---------------------------------------------------------------------------

def test_format_zoo_shows_provenance_as_a_column(zoo_tree):
    text = zoo.format_zoo(zoo.discover_local(zoo_tree))
    assert "trained on" in text
    assert "/data/plaques/train" in text
    assert "unknown" in text
    assert "no checksum on record" in text


def test_format_zoo_on_nothing_says_what_to_do():
    text = zoo.format_zoo([])
    assert "nothing found" in text
    assert "discover_local" in text


def test_format_benchmarks_on_nothing_is_not_a_crash():
    assert zoo.format_benchmarks([]) == "No benchmark to show."


def test_human_bytes_never_prints_a_bare_zero():
    assert zoo._human_bytes(0) == "unknown"
    assert zoo._human_bytes(None) == "unknown"
    assert zoo._human_bytes(2048) == "2.0 KB"


# ---------------------------------------------------------------------------
# 11. reuse of the existing downloader
# ---------------------------------------------------------------------------

def test_the_bulk_pack_download_delegates_to_the_existing_downloader(monkeypatch):
    """``spacr.utils.download_models`` is the downloader spaCR already ships and
    that ``analyze_plaques`` depends on; the zoo wraps it rather than growing a
    second one. Stubbed here so the test stays offline and torch-free."""
    seen = {}

    def fake_download_models(**kwargs):
        seen.update(kwargs)
        return "/pkg/resources/models"

    monkeypatch.setattr(zoo, "_bulk_downloader",
                        lambda: fake_download_models)
    assert zoo.download_bundled_models(retries=2, delay=0) == \
        "/pkg/resources/models"
    assert seen == {"retries": 2, "delay": 0}


def test_the_package_model_root_is_where_download_models_writes():
    root = zoo.package_model_root()
    assert root.name == "models"
    assert root.parent.name == "resources"
    assert root.parent.parent.name == "spacr"


def test_hf_uri_matches_the_url_the_shipped_downloader_builds():
    assert zoo.hf_uri("me/models", "a.CP_model") == (
        "https://huggingface.co/datasets/me/models/resolve/main/"
        "a.CP_model?download=true")


# ---------------------------------------------------------------------------
# 12. thread-safety of the pieces the GUI runs off-thread
# ---------------------------------------------------------------------------

def test_two_concurrent_fetches_into_one_folder_do_not_collide(remote, tmp_path):
    """The GUI can start a second download before the first retires; the temp
    file is per-download and the versioning is by ``exists()``, so the two must
    not land on the same name or corrupt each other."""
    entry, _source, digest = remote
    dest = tmp_path / "dest"
    dest.mkdir()
    out = []
    errors = []

    def grab():
        try:
            out.append(zoo.fetch(entry, dest))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=grab) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors
    assert len({p.name for p in out}) == 2
    assert sorted(p.name for p in dest.iterdir()) == \
        sorted(p.name for p in out)
    for path in out:
        assert zoo.sha256_file(path) == digest
