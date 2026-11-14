"""Import's "Load test data" button on another screen: a press, then the variant or why not.

``_import_test_data_button`` stores the key, the apply callback and the
reporter on its owner; ``_load_import_variant`` then opens the variant at once
when it is on disk, or downloads it and hands it over (or explains the
failure) when the download is done.  The download itself is replaced through
the ``ask`` hook, so these run without a network.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import import_demo  # noqa: E402

pytestmark = pytest.mark.qt


def _owner(qtbot, tmp_path, monkeypatch, *, present, with_reporter=True):
    """A widget carrying the button, with ``is_present`` answering ``present`` in turn."""
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.applied, widget.said = [], []
    answers = iter(present)
    monkeypatch.setattr(import_demo.ix, "import_example_folder",
                        lambda plate: tmp_path / "examples")
    monkeypatch.setattr(import_demo.ix, "variant_inputs",
                        lambda root, key: {"root": str(root), "key": key})
    monkeypatch.setattr(import_demo.ix, "is_present",
                        lambda root, key: next(answers))
    monkeypatch.setattr(import_demo, "explain_download_failure",
                        lambda exc: f"explained: {exc}")
    button = import_demo._import_test_data_button(
        widget, "tiff", widget.applied.append,
        say=widget.said.append if with_reporter else None)
    return widget, button


def _failure(detail):
    return import_demo.tr("The test data could not be downloaded: {detail}",
                          detail=detail)


def test_load_import_variant_hands_over_the_variant_or_says_why_not(
        qtbot, tmp_path, monkeypatch, caplog, subtests):
    """Each download outcome gives the exact hand-over, report and button state."""
    inputs = {"root": str(tmp_path / "examples"), "key": "tiff"}
    idle = import_demo.tr("Load test data…")
    fetching = import_demo.tr("Fetching test data…")

    def finished(widget, plate, done):
        done(object(), None)

    def failed(widget, plate, done):
        done(None, "disk full")

    def unexplained(widget, plate, done):
        done(None, None)

    def raises(widget, plate, done):
        raise OSError("offline")

    # case, is_present answers, ask, reporter?, returned, applied, said
    cases = [
        ("already on disk", [True], None, True, True, [inputs], []),
        ("download finished", [False, True], finished, True, False,
         [inputs], []),
        ("download finished without the variant", [False, False], finished,
         True, False, [], [_failure(import_demo.tr("unknown error"))]),
        ("download failed", [False], failed, True, False, [],
         [_failure("disk full")]),
        ("download raised", [False], raises, True, False, [],
         [_failure("explained: offline")]),
    ]
    for case, present, ask, reporter, returned, applied, said in cases:
        with subtests.test(msg=case):
            owner, button = _owner(qtbot, tmp_path, monkeypatch,
                                   present=present, with_reporter=reporter)
            states = []
            if ask is None:
                def ask_hook(*_args):
                    pytest.fail("a variant on disk must not be downloaded")
            else:
                def ask_hook(widget, plate, done, _ask=ask):
                    assert widget is owner and plate == tmp_path
                    states.append((button.isEnabled(), button.text()))
                    _ask(widget, plate, done)
            got = import_demo._load_import_variant(owner, ask=ask_hook,
                                                   plate=tmp_path)
            assert got is returned
            assert owner.applied == applied
            assert owner.said == said
            assert states == ([] if ask is None else [(False, fetching)])
            assert (button.isEnabled(), button.text()) == (True, idle)

    with subtests.test(msg="no reporter: the failure is logged"):
        owner, button = _owner(qtbot, tmp_path, monkeypatch, present=[False],
                               with_reporter=False)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=import_demo.LOG.name):
            assert import_demo._load_import_variant(
                owner, ask=unexplained, plate=tmp_path) is False
        messages = [r.getMessage() for r in caplog.records
                    if r.name == import_demo.LOG.name]
        assert messages == [_failure(import_demo.tr("unknown error"))]
        assert owner.applied == [] and owner.said == []

    with subtests.test(msg="the button press loads the owner's variant"):
        owner, button = _owner(qtbot, tmp_path, monkeypatch, present=[True])
        monkeypatch.setattr(import_demo, "example_plate_folder",
                            lambda: tmp_path)
        button.click()
        assert owner.applied == [inputs]
        assert owner._import_test_data[1] == "tiff"
