"""The "Load test data" routes: what the button, the console and src show.

Pins, for each example download an AppScreen offers (the published screen's
databases and crops, Measure's plate, the annotation plate and the
sequencing reads):

* the button goes to "Fetching..." and disabled while the download runs,
  and comes back enabled with its own caption whether it worked or failed;
* a failed or cancelled download is named in the console and changes no
  setting;
* a finished download says where the data is and points ``src`` at it;
* with no picker/downloader passed, the real one is the one reached for;
* a screen with no ``src`` control, no console, or no button still answers
  (the fallback folder, ``False`` or the placed folder).
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QPushButton  # noqa: E402

from spacr.qt import hf_download  # noqa: E402
from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402

pytestmark = pytest.mark.qt


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.text()
                     for block in console.findChildren(_StdoutBlock))


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


@pytest.fixture
def plate(tmp_path, monkeypatch):
    folder = tmp_path / "plate"
    monkeypatch.setattr(hf_download, "example_plate_folder", lambda: folder)
    return folder


def _src(screen):
    return screen._settings_model._widgets["src"].text()


# --------------------------------------------------------------------------
# the published screen's measurements and crops


class TestTheScreenData:

    def test_the_destination_is_the_shared_plate_folder(self, screen, plate):
        assert screen.screen_data_destination() == plate

    def test_nothing_chosen_fetches_nothing(self, screen, plate):
        asked = []
        placed = screen.load_the_screen_data(
            kind="measurements", choose=lambda *a: [],
            ask=lambda *a: asked.append(a))
        assert placed == {}
        assert asked == []
        assert plate.is_dir()

    def test_a_failed_fetch_restores_the_button_and_says_why(
            self, screen, plate):
        button = QPushButton("Measurements (.db)")
        screen._screen_feature_button = button
        chosen = [types.SimpleNamespace(archive="plate1.tar")]
        seen = {}

        def download(_screen, destination, archives, repo, done):
            seen.update(enabled=button.isEnabled(), text=button.text(),
                        archives=list(archives), repo=repo)
            done(None, "no route to huggingface.co")

        placed = screen.load_the_screen_data(
            kind="measurements", choose=lambda *a: chosen, ask=download)
        assert placed == {}
        assert seen["enabled"] is False
        assert "Fetching" in seen["text"]
        assert seen["archives"] == ["plate1.tar"]
        assert seen["repo"]
        assert button.isEnabled()
        assert button.text() == "Feature"
        assert "no route to huggingface.co" in _console_text(screen._console)

    def test_a_cancelled_fetch_says_cancelled(self, screen, plate):
        button = QPushButton("Image crops")
        screen._screen_crops_button = button
        screen.load_the_screen_data(
            kind="crops",
            choose=lambda *a: [types.SimpleNamespace(archive="c.tar")],
            ask=lambda _s, _d, _a, _r, done: done(None, None))
        assert button.isEnabled()
        assert button.text() == "Image crops"
        assert "cancelled" in _console_text(screen._console)

    def test_a_finished_fetch_names_the_folder(self, screen, plate):
        screen.__dict__.pop("_screen_data_button", None)
        placed = screen.load_the_screen_data(
            choose=lambda *a: [types.SimpleNamespace(archive="a.tar")],
            ask=lambda _s, _d, _a, _r, done: done(object(), None))
        assert placed == {"src": str(plate)}
        assert f"Screen data ready: {plate}" in _console_text(screen._console)

    def test_the_real_picker_and_downloader_are_reached_for(
            self, screen, plate, monkeypatch):
        from spacr.qt.widgets import screen_data_picker

        picked = []
        fetched = []
        monkeypatch.setattr(
            screen_data_picker, "choose_screen_data",
            lambda parent, destination, kind: picked.append(kind) or [
                types.SimpleNamespace(archive="x.tar")])
        monkeypatch.setattr(
            hf_download, "download_chosen_screen_data",
            lambda parent, destination, archives, repo, done:
                fetched.append(list(archives)))
        assert screen.load_the_screen_data(kind="measurements") == {}
        assert picked == ["measurements"]
        assert fetched == [["x.tar"]]


# --------------------------------------------------------------------------
# Measure's plate


class TestTheMeasurePlate:

    def test_a_failed_download_leaves_src_and_restores_the_button(
            self, screen, plate):
        button = QPushButton("Load test data…")
        screen._measure_example_button = button
        before = _src(screen)
        seen = []

        def download(_screen, destination, done):
            seen.append((button.isEnabled(), button.text()))
            done(None, "disk full")

        assert screen.load_the_measure_example(ask=download) == {}
        assert seen == [(False, "Fetching test data…")]
        assert button.isEnabled()
        assert button.text() == "Load test data…"
        assert _src(screen) == before
        assert "disk full" in _console_text(screen._console)

    def test_a_finished_download_points_src_at_the_plate(
            self, screen, plate, monkeypatch):
        screen._measure_example_button = None
        placed = screen.load_the_measure_example(
            ask=lambda _s, destination, done: done(destination, None))
        assert placed == {"src": str(plate)}
        assert _src(screen) == str(plate)

    def test_the_real_downloader_is_reached_for(self, screen, plate,
                                                monkeypatch):
        called = []
        monkeypatch.setattr(hf_download, "download_measure_example",
                            lambda parent, destination, done:
                                called.append(destination))
        screen._measure_example_button = None
        assert screen.load_the_measure_example() == {}
        assert called == [plate]

    def test_the_example_goes_in_even_where_src_cannot_be_written(
            self, screen, plate):
        plate.mkdir(parents=True)
        was = screen._settings_model
        screen._settings_model = types.SimpleNamespace(
            _widgets={"src": types.SimpleNamespace()})
        try:
            placed = screen._put_the_measure_example_in_place(plate)
        finally:
            screen._settings_model = was
        assert placed == {"src": str(plate)}
        assert f"Source directory (src): {plate}" in _console_text(
            screen._console)

    def test_a_screen_without_src_keeps_the_fallback(self, screen, tmp_path):
        was = screen._settings_model
        screen._settings_model = types.SimpleNamespace(_widgets={})
        try:
            assert screen.keep_the_src_openable(tmp_path) == str(tmp_path)
        finally:
            screen._settings_model = was


# --------------------------------------------------------------------------
# pointing src at a folder


class TestPointingSrcAtAFolder:

    def test_a_screen_without_src_says_it_did_not_take(self, screen,
                                                       tmp_path):
        was = screen._settings_model
        screen._settings_model = types.SimpleNamespace(_widgets={})
        try:
            assert screen.point_src_at(tmp_path) is False
        finally:
            screen._settings_model = was
        assert _src(screen) != str(tmp_path)

    def test_without_a_console_src_is_still_set(self, screen, tmp_path):
        was = screen._console
        screen._console = None
        try:
            assert screen.point_src_at(tmp_path) is True
        finally:
            screen._console = was
        assert _src(screen) == str(tmp_path)


# --------------------------------------------------------------------------
# sequencing reads


class TestTheSequencingReads:

    def test_the_real_picker_is_opened_and_its_files_are_used(
            self, screen, plate, monkeypatch):
        from spacr.qt.widgets import sra_picker

        opened = []

        class _Picker:
            def __init__(self, destination, parent):
                self.destination = destination
                self.written = []

            def exec(self):
                opened.append(self.destination)
                self.written = ["a_1.fastq.gz", "a_2.fastq.gz"]

        monkeypatch.setattr(sra_picker, "SraPicker", _Picker)
        placed = screen.load_the_sequencing_example()
        folder = plate.parent / "sequencing"
        assert opened == [folder]
        assert placed == {"src": str(folder),
                          "files": ["a_1.fastq.gz", "a_2.fastq.gz"]}
        assert _src(screen) == str(folder)
        assert f"2 read files ready: {folder}" in _console_text(
            screen._console)

    def test_a_picker_with_no_dialog_and_nothing_written_changes_nothing(
            self, screen, plate):
        before = _src(screen)
        assert screen.load_the_sequencing_example(
            picker=types.SimpleNamespace(written=[])) == {}
        assert _src(screen) == before

    def test_reads_arrive_on_a_screen_without_src(self, screen, plate):
        was = screen._settings_model
        screen._settings_model = types.SimpleNamespace(_widgets={})
        try:
            placed = screen.load_the_sequencing_example(
                picker=types.SimpleNamespace(written=["r.fastq"]))
        finally:
            screen._settings_model = was
        assert placed["files"] == ["r.fastq"]
        assert "1 read files ready" in _console_text(screen._console)


# --------------------------------------------------------------------------
# the annotation plate


class TestTheAnnotationPlate:

    def test_a_failed_download_restores_the_button_and_says_why(
            self, screen, plate):
        button = QPushButton("Load test data…")
        screen._annotate_example_button = button
        seen = []

        def download(_screen, destination, done):
            seen.append((button.isEnabled(), button.text()))
            done(None, "")

        assert screen.load_the_annotate_example(ask=download) == {}
        assert seen == [(False, "Fetching test data…")]
        assert button.isEnabled()
        assert button.text() == "Load test data…"
        assert "cancelled" in _console_text(screen._console)

    def test_a_finished_download_applies_what_came_with_it(
            self, screen, plate, monkeypatch):
        screen.__dict__.pop("_annotate_example_button", None)
        applied_from = []
        monkeypatch.setattr(
            screen, "_apply_the_example_settings",
            lambda destination: applied_from.append(destination) or {
                "src": str(destination)})
        placed = screen.load_the_annotate_example(
            ask=lambda _s, destination, done: done(destination, None))
        assert applied_from == [plate]
        assert placed == {"src": str(plate)}

    def test_the_real_downloader_is_reached_for(self, screen, plate,
                                                monkeypatch):
        called = []
        monkeypatch.setattr(hf_download, "download_annotate_example",
                            lambda parent, destination, done:
                                called.append(destination))
        screen.__dict__.pop("_annotate_example_button", None)
        assert screen.load_the_annotate_example() == {}
        assert called == [plate]
