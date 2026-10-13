"""Import examples reuse the right converter and report every download ending."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QWidget

from spacr.qt import import_demo as demo
from spacr.qt.screens.convert import ConvertScreen


@pytest.fixture(autouse=True)
def no_real_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the test attempted to start a real download thread")
    monkeypatch.setattr(demo, "download_toxo_mito_demo", forbidden)


@pytest.fixture
def converter(qtbot):
    widget = ConvertScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_download_adapter_selects_import_archive_worker(tmp_path, monkeypatch):
    download = Mock()
    monkeypatch.setattr(demo, "download_toxo_mito_demo", download)
    parent, finished = object(), Mock()
    demo.download_import_example(parent, str(tmp_path), finished)
    download.assert_called_once_with(
        parent, tmp_path, finished, worker_factory=demo._ImportTarWorker,
        title="Downloading the Import test data")


@pytest.mark.parametrize("error,expected,is_error", [
    (demo.CANCELLED, "cancelled", False),
    ("connection lost", "connection lost", True),
])
def test_download_without_a_button_still_reports_its_outcome(
        tmp_path, monkeypatch, error, expected, is_error):
    screen = SimpleNamespace(_set_status=Mock())
    pending = []
    monkeypatch.setattr(demo, "example_plate_folder", lambda: tmp_path)
    def start(parent, plate, on_done):
        assert parent is screen and plate == tmp_path
        pending.append(on_done)
    monkeypatch.setattr(demo, "download_import_example", start)
    assert demo.load_import_test_data(screen, "zeiss_czi") is False
    assert "Downloading" in screen._set_status.call_args.args[0]
    assert len(pending) == 1
    pending[0](None, error)
    call = screen._set_status.call_args
    assert expected in call.args[0]
    assert call.kwargs.get("error", False) is is_error


def test_destination_skips_both_existing_files_and_folders(tmp_path):
    images = tmp_path / "images"
    (tmp_path / "images_spacr").mkdir()
    (tmp_path / "images_spacr_2").write_text("existing project marker")
    assert demo._fresh_destination(str(images) + "/") == str(tmp_path / "images_spacr_3")
    assert not (tmp_path / "images_spacr_3").exists()
    assert (tmp_path / "images_spacr_2").read_text() == "existing project marker"


@pytest.mark.parametrize("wrapped", [False, True])
def test_fold_opener_reuses_the_existing_converter(converter, qtbot, wrapped):
    if wrapped:
        wrapper = QWidget()
        qtbot.addWidget(wrapper)
        converter.setParent(wrapper)
        shown = wrapper
    else:
        shown = converter
    irrelevant = Mock()
    chosen = Mock(return_value=shown)
    screen = SimpleNamespace(_fold_openers=[
        SimpleNamespace(key="other", open=irrelevant),
        SimpleNamespace(key="convert", open=chosen),
    ])
    assert demo._converter_for(screen) is converter
    chosen.assert_called_once_with()
    irrelevant.assert_not_called()
    assert not hasattr(screen, "_test_converter")
    # qtbot owns both widgets; release Qt's parent ownership before teardown.
    if wrapped:
        converter.setParent(None)


def test_missing_fold_pages_reuse_cached_standalone_converter(converter, qtbot):
    empty = QWidget()
    qtbot.addWidget(empty)
    screen = SimpleNamespace(_fold_openers=[
        SimpleNamespace(key="convert", open=lambda: None),
        SimpleNamespace(key="convert", open=lambda: empty),
    ], _test_converter=converter)
    assert demo._converter_for(screen) is converter


def test_without_a_fold_strip_one_standalone_converter_is_opened(qtbot, monkeypatch):
    opened = []
    def show(widget, parent, title):
        qtbot.addWidget(widget)
        opened.append((widget, parent, title))
    monkeypatch.setattr("spacr.qt.screens.map_barcodes.show_as_window", show)
    screen = SimpleNamespace()
    first = demo._converter_for(screen)
    assert isinstance(first, ConvertScreen)
    assert demo._converter_for(screen) is first
    assert opened == [(first, screen, "Format Converter")]


def test_unavailable_converter_is_an_explicit_error(monkeypatch):
    screen = SimpleNamespace(_set_status=Mock())
    monkeypatch.setattr(demo, "_converter_for", lambda parent: None)
    assert demo._apply_to_converter(screen, {"images": "unused"}) is False
    screen._set_status.assert_called_once_with(
        "The Format Converter could not be opened.", error=True)


def test_converter_preview_refusal_is_returned_without_importing(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "images_yokogawa").mkdir()
    converter = SimpleNamespace(set_source=Mock(), set_destination=Mock(),
                                preview=Mock(return_value=False))
    screen = SimpleNamespace(_set_status=Mock())
    monkeypatch.setattr(demo, "_converter_for", lambda parent: converter)
    assert demo._apply_to_converter(screen, {"images": images}) is False
    converter.set_source.assert_called_once_with(str(images))
    converter.set_destination.assert_called_once_with(str(tmp_path / "images_yokogawa_2"))
    converter.preview.assert_called_once_with()
    assert str(images) in screen._set_status.call_args.args[0]
