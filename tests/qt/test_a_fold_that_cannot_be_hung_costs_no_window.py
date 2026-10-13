"""Every part of a fold is optional; none of them may cost the window.

A folded module reaches its host through three pieces hung on from
outside: the strip on the masthead, the icon beside the settings
categories that came with it, and the preview panel it brought. Each is
built after the screen is, and each failure below leaves the screen
standing rather than taking it down with it.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMainWindow, QWidget  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_fold_strip_hook_that_raises_leaves_the_window_built(
        qapp, monkeypatch, caplog):
    from spacr.qt import shortcuts as sc
    from spacr.qt.screens import map_barcodes

    def refuse(_window):
        raise RuntimeError("no masthead on this window")

    monkeypatch.setattr(map_barcodes, "install_window_hooks", refuse)

    window = QMainWindow()
    try:
        with caplog.at_level(logging.DEBUG, logger=sc.LOG.name):
            sc._install_window_hooks(window)
    finally:
        window.close()
        window.deleteLater()

    assert any("fold-strip hooks" in record.message
               for record in caplog.records), (
        "the failure is recorded rather than raised")


def test_a_category_mark_that_raises_marks_nothing_and_says_nothing(
        qapp, monkeypatch, caplog):
    from spacr.qt.screens import measure

    def refuse(_sections, _folds):
        raise RuntimeError("this screen has no categories yet")

    monkeypatch.setattr(measure, "mark_folded_categories", refuse)

    screen = QWidget()
    screen.app_key = measure.HOST_KEY
    screen._settings_sections = ()

    with caplog.at_level(logging.DEBUG, logger=measure.LOG.name):
        assert measure.mark_fold_sources(screen) == {}

    assert any("folded categories" in record.getMessage()
               for record in caplog.records)


def test_a_screen_that_is_not_the_host_marks_nothing(qapp):
    from spacr.qt.screens import measure

    screen = QWidget()
    screen.app_key = "sequencing"

    assert measure.mark_fold_sources(screen) == {}


def test_a_preview_whose_builder_cannot_be_found_is_not_remembered(qapp):
    from spacr.qt import preview_registry as registry

    spec = registry.PreviewSpec(builder="no.such.module:build_nothing",
                                title="Nowhere preview")
    screen = QWidget()

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(registry.PREVIEWS, "nowhere", spec)
        assert registry.attach_folded(screen, "nowhere") is None

    assert screen._folded_previews == {}, (
        "nothing was attached, so nothing may be cached as attached")


def test_a_key_that_declares_no_preview_attaches_nothing(qapp):
    from spacr.qt import preview_registry as registry

    screen = QWidget()

    assert registry.attach_folded(screen, "not-a-module") is None
    assert screen._folded_previews == {}
