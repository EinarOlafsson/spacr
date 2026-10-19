"""Item 560: which DINO the screen runs, and the cell example's late answers.

The "DINOv2 (Cell-DINO when installed)" choice runs a configured Cell-DINO
checkpoint once its planes are mapped, otherwise the open DINOv2, and says
why on the alpha line, including after an embedding finishes. The Cell-DINO
form fills itself from an installed checkpoint it finds. The annotated cell
example downloads the plate when it is not on disk and says why when that
fails; a load still running, a stop or a closed screen keeps a second or
late answer from loading anything.

Real screen throughout; only the background runner (to hold an answer back),
the plate download and the encoder are replaced.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QDialog

from spacr.qt.screens import embeddings as em

from .test_embedding_synthetic_demo import _fake_example_plate

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, monkeypatch, tmp_path):
    from spacr import embeddings
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    monkeypatch.delenv("SPACR_CELL_DINO_CHECKPOINT", raising=False)
    monkeypatch.setenv("SPACR_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setattr(embeddings, "_CELL_DINO_SEEN", {})
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class Held:
    """A runner stand-in that keeps each submission instead of running it."""

    def __init__(self):
        self.calls = []

    def __call__(self, work, done):
        self.calls.append((work, done))


def _auto(screen):
    screen._foundation.setCurrentIndex(screen._foundation.findData("dino_auto"))


def test_a_configured_mapped_cell_dino_checkpoint_is_the_one_that_runs(screen, tmp_path):
    checkpoint = tmp_path / "cp_vits8.pth"
    checkpoint.write_bytes(b"weights are never read by this choice")
    screen.set_crops(np.zeros((2, 8, 8, 5), dtype=np.float32))
    screen._cell_dino_factory = "cell_dino_cp_vits8"
    screen._cell_dino_channels = (4, 2, 0, 3, 1)
    screen._cell_dino_checkpoint = str(checkpoint)
    screen._cell_dino_digest = "ab" * 32
    _auto(screen)

    spec, note = screen._dino_note()
    assert (spec.backbone, spec.cell_dino_factory, spec.channels) == (
        "cell_dino", "cell_dino_cp_vits8", (4, 2, 0, 3, 1))
    assert spec.checkpoint_path == str(checkpoint)
    assert spec.checkpoint_sha256 == "ab" * 32
    assert note == {"model": "cell_dino", "factory": "cell_dino_cp_vits8",
                    "path": str(checkpoint)}
    assert screen._dino_in_use.text() == (
        "In use: official Cell-DINO (cell_dino_cp_vits8), checkpoint "
        "cp_vits8.pth · abababababab….")
    assert screen.spec().backbone == "cell_dino"

    screen._cell_dino_channels = (4, 2, 0, 3)
    _auto(screen)
    assert screen.spec().backbone == "vit_small_patch14_dinov2.lvd142m"


def test_a_checkpoint_path_that_is_not_a_file_is_named_as_unreadable(
        screen, monkeypatch, tmp_path):
    folder = tmp_path / "not_a_model"
    folder.mkdir()
    monkeypatch.setenv("SPACR_CELL_DINO_CHECKPOINT", str(folder))
    screen.set_crops(np.zeros((2, 8, 8, 3), dtype=np.float32))
    _auto(screen)

    text = screen._dino_in_use.text()
    assert text.startswith("In use: open DINOv2 ViT-S/14 (general-image weights). ")
    assert text.endswith(f"{folder} is not a readable official Cell-DINO model state.")
    assert screen.spec().backbone == "vit_small_patch14_dinov2.lvd142m"


def test_the_cell_dino_form_starts_from_an_installed_checkpoint_it_finds(
        screen, monkeypatch, tmp_path):
    from spacr import embeddings

    found = tmp_path / "hpa.pth"
    found.write_bytes(b"x")
    asked = []
    monkeypatch.setattr(embeddings, "_find_cell_dino", lambda verify=False, **choice: (
        asked.append(verify) or chose.append(choice) or {
            "path": str(found), "factory": "cell_dino_hpa_vitl16", "sha256": "c" * 64}))
    chose = []
    seen = []
    monkeypatch.setattr(QDialog, "exec", lambda dialog: seen.append(
        dialog.objectName()) or QDialog.Rejected)
    screen.set_crops(np.zeros((2, 8, 8, 4), dtype=np.float32))

    screen._choose_cell_dino()

    assert asked == [True] and seen == ["EmbeddingsCellDinoDialog"]
    assert chose == [{"available": 4, "preferred": ""}]
    assert (screen._cell_dino_factory, screen._cell_dino_checkpoint,
            screen._cell_dino_digest) == ("cell_dino_hpa_vitl16", str(found), "c" * 64)

    monkeypatch.setattr(embeddings, "_find_cell_dino", lambda verify=False, **_: (
        asked.append(verify) or {"path": str(found), "factory": "cell_dino_cp_vits8",
                                 "sha256": "d" * 64}))
    screen._choose_cell_dino()
    assert asked == [True]
    assert screen._cell_dino_factory == "cell_dino_hpa_vitl16"


def test_the_cell_dino_form_without_an_installed_checkpoint_keeps_nothing(
        screen, monkeypatch):
    from spacr import embeddings

    monkeypatch.setattr(embeddings, "_find_cell_dino",
                        lambda verify=False, **_: {"path": "/x.pth", "problem": "missing"})
    monkeypatch.setattr(QDialog, "exec", lambda dialog: QDialog.Rejected)
    screen._choose_cell_dino()
    assert not screen._cell_dino_checkpoint and not screen._cell_dino_digest


def test_an_auto_embedding_says_which_dino_ran_on_the_line_and_the_status(
        screen, monkeypatch):
    from spacr import embeddings

    monkeypatch.setattr(embeddings, "encoder_entry", lambda spec: SimpleNamespace(
        name=spec.backbone, sha256=""))
    ran = []

    def encode(crops, spec, *, record):
        ran.append(spec.backbone)
        values = crops.mean(axis=(1, 2)).astype(np.float32)
        return embeddings.EmbeddingResult(values, ("emb_0", "emb_1", "emb_2"), spec, 3, 3)

    monkeypatch.setattr(embeddings, "_embed_plate", encode)
    screen.set_crops(np.arange(4 * 6 * 6 * 3, dtype=np.float32).reshape(4, 6, 6, 3))
    _auto(screen)
    screen._dino_in_use.setText("")

    screen.embed()

    assert ran == ["vit_small_patch14_dinov2.lvd142m"]
    sentence = screen._dino_in_use.text()
    assert sentence.startswith("In use: open DINOv2 ViT-S/14")
    assert "No official Cell-DINO checkpoint is installed" in sentence
    assert screen._status.text().endswith(" " + sentence)
    assert screen._result is not None


def _cell_plate(monkeypatch, tmp_path):
    from spacr import example_archives
    from spacr.qt.widgets import measurements_example

    database = tmp_path / "plate1" / "measurements" / "measurements.db"
    monkeypatch.setattr(measurements_example, "example_measurements_db", lambda: database)
    monkeypatch.setattr(measurements_example, "example_measurements_folder",
                        lambda: tmp_path / "plate1")
    monkeypatch.setattr(example_archives, "example_plate_folder",
                        lambda: tmp_path / "plate1")
    return database


def test_the_cell_example_is_downloaded_first_and_says_why_it_could_not_be(
        screen, monkeypatch, tmp_path):
    from spacr.qt import hf_download

    database = _cell_plate(monkeypatch, tmp_path)
    asked = []
    monkeypatch.setattr(hf_download, "download_annotate_example",
                        lambda parent, dest, done: asked.append((parent, dest, done)))
    button = screen._cell_example_button

    button.click()

    assert [(parent, dest) for parent, dest, _done in asked] == [(screen, tmp_path / "plate1")]
    assert not button.isEnabled()
    assert screen._status.text() == "Fetching test data…"
    done = asked[0][2]
    done(None, "the network is down")
    assert button.isEnabled()
    assert screen._status.text() == "The test data could not be downloaded: the network is down"
    done(SimpleNamespace(), "")
    assert screen._status.text() == "The test data could not be downloaded: unknown error"
    assert getattr(screen, "_crops", None) is None

    _fake_example_plate(tmp_path / "plate1")
    assert database.is_file()
    done(SimpleNamespace(), "")
    assert screen._crops.shape[0] == 5
    assert list(screen._labels.values()) == ["1", "2", "2", "1", "2"]
    assert screen._foundation.currentData() == "dino_auto"


def test_a_cell_example_already_loading_or_on_a_closed_screen_starts_nothing(
        screen, monkeypatch, tmp_path):
    _cell_plate(monkeypatch, tmp_path)
    _fake_example_plate(tmp_path / "plate1")
    held = Held()
    monkeypatch.setattr(screen._jobs, "submit", held)

    screen._load_cell_example()
    screen._load_cell_example()
    assert len(held.calls) == 1 and screen.is_loading()
    assert screen._status.text() == "Loading the annotated example cells…"

    screen.close()
    screen._loading = False
    screen._load_cell_example()
    assert len(held.calls) == 1


def test_a_stopped_cell_example_is_headed_as_a_stop_and_loads_nothing(
        screen, monkeypatch, tmp_path):
    _cell_plate(monkeypatch, tmp_path)
    _fake_example_plate(tmp_path / "plate1")
    held = Held()
    monkeypatch.setattr(screen._jobs, "submit", held)
    screen._load_cell_example()
    work, done = held.calls[0]
    answer = work()
    screen._stop.set()

    done(answer)

    assert not screen.is_loading()
    assert screen._status.text() == "Loading the annotated example cells was stopped."
    assert getattr(screen, "_crops", None) is None


def test_a_cell_example_finishing_after_close_is_not_loaded(
        screen, monkeypatch, tmp_path):
    _cell_plate(monkeypatch, tmp_path)
    _fake_example_plate(tmp_path / "plate1")
    held = Held()
    monkeypatch.setattr(screen._jobs, "submit", held)
    screen._load_cell_example()
    work, done = held.calls[0]
    answer = work()
    screen.close()

    done(answer)

    assert getattr(screen, "_crops", None) is None
    assert not screen._labels
