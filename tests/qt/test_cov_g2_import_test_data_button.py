"""Import's Load test data button, from a click to the variant or an error."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import import_demo  # noqa: E402


@pytest.fixture
def owner(qtbot, tmp_path, monkeypatch):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.applied, widget.said = [], []
    monkeypatch.setattr(import_demo.ix, "import_example_folder",
                        lambda plate: tmp_path / "examples")
    monkeypatch.setattr(import_demo.ix, "variant_inputs",
                        lambda root, key: {"root": str(root), "key": key})
    import_demo._import_test_data_button(
        widget, "tiff", widget.applied.append, say=widget.said.append)
    return widget


def test_a_variant_on_disk_is_handed_over_at_once(owner, monkeypatch,
                                                   tmp_path):
    monkeypatch.setattr(import_demo.ix, "is_present", lambda root, key: True)
    assert import_demo._load_import_variant(owner, plate=tmp_path) is True
    assert owner.applied == [{"root": str(tmp_path / "examples"),
                              "key": "tiff"}]


def test_a_finished_download_hands_over_the_variant(owner, monkeypatch,
                                                     tmp_path):
    present = iter([False, True])
    monkeypatch.setattr(import_demo.ix, "is_present",
                        lambda root, key: next(present))
    button = owner._import_test_data[0]

    def ask(widget, plate, done):
        assert not button.isEnabled()
        done(object(), None)

    assert import_demo._load_import_variant(owner, ask=ask,
                                            plate=tmp_path) is False
    assert button.isEnabled()
    assert owner.applied and owner.said == []


def test_a_failed_download_says_why(owner, monkeypatch, tmp_path):
    monkeypatch.setattr(import_demo.ix, "is_present", lambda root, key: False)
    import_demo._load_import_variant(
        owner, ask=lambda w, p, done: done(None, "disk full"), plate=tmp_path)
    assert owner.applied == []
    assert owner.said and "disk full" in owner.said[0]


def test_a_download_that_raises_is_explained(owner, monkeypatch, tmp_path):
    monkeypatch.setattr(import_demo.ix, "is_present", lambda root, key: False)
    monkeypatch.setattr(import_demo, "explain_download_failure",
                        lambda exc: f"explained: {exc}")

    def ask(widget, plate, done):
        raise OSError("offline")

    import_demo._load_import_variant(owner, ask=ask, plate=tmp_path)
    assert owner.said == [owner.said[0]]
    assert "explained: offline" in owner.said[0]


def test_without_a_reporter_a_failure_is_logged(qtbot, tmp_path, monkeypatch,
                                                caplog):
    widget = QWidget()
    qtbot.addWidget(widget)
    monkeypatch.setattr(import_demo.ix, "import_example_folder",
                        lambda plate: tmp_path)
    monkeypatch.setattr(import_demo.ix, "is_present", lambda root, key: False)
    import_demo._import_test_data_button(widget, "tiff", lambda inputs: None)
    with caplog.at_level("INFO", logger=import_demo.LOG.name):
        import_demo._load_import_variant(
            widget, ask=lambda w, p, done: done(None, None), plate=tmp_path)
    assert any("could not be downloaded" in r.getMessage()
               for r in caplog.records)
