"""The Embeddings screen's own choosers, refusals and teardown.

Pinned here, each as what the user sees or gets:

* the Browse button asks for a database file or for a folder, depending on
  where the crops come from, fills the path box with the answer and offers
  Load; a dismissed dialog leaves the box empty;
* pressing Load while a load runs stops it; pressing it with no path says
  what to choose instead of trying;
* the loaded sentence says when a load was stopped early and when crops
  were padded or trimmed, and to what size;
* the backbone list offers each network once, however often it is filled;
* closing the screen stops its load even when its job runner cannot be
  shut down cleanly;
* the app registry is handed the screen's factory, and the factory makes
  the screen.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.crop_loader import CROP_SOURCE_DATABASE, CROP_SOURCE_FOLDER
from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _source(screen, which):
    screen._where.setCurrentIndex(screen._where.findData(which))
    assert screen.crop_source() == which


# ---------------------------------------------------------------------------
# Browse
# ---------------------------------------------------------------------------

def test_browse_asks_for_a_database_file(screen, tmp_path, monkeypatch):
    asked = []
    chosen = str(tmp_path / "measurements.db")

    def open_file(parent, title, start, filters):
        asked.append((title, filters))
        return chosen, filters

    monkeypatch.setattr(em.QFileDialog, "getOpenFileName",
                        staticmethod(open_file))
    _source(screen, CROP_SOURCE_DATABASE)

    assert screen.choose_path() == chosen
    assert asked and "database" in asked[0][0]
    assert screen._path.text() == chosen
    assert screen._load.isEnabled()


def test_browse_asks_for_a_folder_of_crops(screen, tmp_path, monkeypatch):
    asked = []

    def pick_folder(parent, title, start):
        asked.append(title)
        return str(tmp_path)

    monkeypatch.setattr(em.QFileDialog, "getExistingDirectory",
                        staticmethod(pick_folder))
    _source(screen, CROP_SOURCE_FOLDER)

    assert screen.choose_path() == str(tmp_path)
    assert asked and "folder" in asked[0]
    assert screen._path.text() == str(tmp_path)
    assert screen._load.isEnabled()


def test_a_dismissed_chooser_changes_nothing(screen, monkeypatch):
    monkeypatch.setattr(em.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a: ""))
    _source(screen, CROP_SOURCE_FOLDER)

    assert screen.choose_path() == ""
    assert screen._path.text() == ""
    assert not screen._load.isEnabled()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def test_pressing_load_during_a_load_stops_it(screen):
    screen._set_loading(True)
    assert screen._load.text() == "Stop"

    screen.load_crops()
    assert screen._stop.is_set()
    assert screen._status.text() == "Stopping after this page…"


def test_load_with_no_path_says_what_to_choose(screen):
    screen._path.setText("   ")
    screen.load_crops()
    assert not screen.is_loading()
    assert "Choose a measurements database" in screen._status.text()
    assert screen._pages.currentWidget() is screen._state


def test_the_sentence_says_a_load_was_stopped_and_what_was_resized(screen):
    text = screen._loaded_sentence(
        {"loaded": 12, "matched": 40, "stopped": True, "conformed": 3,
         "crop_shape": [64, 64, 3], "source": "png crop source"})
    assert "12 crops loaded" in text
    assert "stopped early; 40 matched" in text
    assert "3 padded or trimmed to 64x64" in text
    assert "raise" not in text


# ---------------------------------------------------------------------------
# Backbones, closing, registering
# ---------------------------------------------------------------------------

def test_the_backbones_are_offered_once_each(screen):
    before = [screen._backbone.itemText(i)
              for i in range(screen._backbone.count())]
    screen._fill_backbones()
    after = [screen._backbone.itemText(i)
             for i in range(screen._backbone.count())]
    assert after == before
    assert len(set(after)) == len(after)


def test_closing_stops_the_load_even_when_the_runner_will_not_shut_down(
        screen, monkeypatch):
    def stuck():
        raise RuntimeError("worker would not stop")

    monkeypatch.setattr(screen._jobs, "shutdown", stuck)
    screen.show()
    assert screen.close()
    assert screen._stop.is_set()
    assert not screen.isVisible()


def test_the_registry_is_handed_the_screen_factory(qtbot, monkeypatch):
    from spacr.qt import app

    handed = []
    monkeypatch.setattr(app, "register_app",
                        lambda key, factory: handed.append((key, factory))
                        or True)
    assert em.register() is True
    assert handed == [(em.APP_KEY, em.make_embeddings_screen)]

    made = em.make_embeddings_screen()
    qtbot.addWidget(made)
    assert isinstance(made, em.EmbeddingsScreen)
