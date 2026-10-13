"""Item 288: Annotate's console report, test-data pack and judgement badge.

Three small pieces of the Annotate screen that each have a way to go wrong
the ordinary tests do not reach:

* **File as issue** can find an empty console, a reporter that will not
  import, a preview the user closes, a report that cannot be built, a
  GitHub that refuses it, or a job runner that will not start. Every one of
  those must end in a sentence in the console, and none may send anything.
  The happy path is pinned in
  ``test_annotate_console_can_be_copied_and_reported.py``.
* **A test-data settings pack** is applied field by field, so a crop size
  that is not a number costs that field and nothing else.
* **The judgement badge** draws a tick, a cross or a question mark; each
  must actually be drawn, and each differently, or a confirmed crop and a
  rejected one look the same.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter           # noqa: E402
from PySide6.QtWidgets import QDialog                        # noqa: E402

from spacr.qt import preferences                             # noqa: E402


@pytest.fixture
def screen(qtbot, monkeypatch):
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget._report_jobs, "_threaded", False)
    preferences.set_issue_prompt_mode("ask")
    return widget


@pytest.fixture
def sent(monkeypatch):
    """Everything that reached GitHub; the test fails if a report did."""
    shipped = []
    monkeypatch.setattr("spacr.qt.ai.issue_report.submit_report",
                        lambda report: shipped.append(report) or "https://x/9")
    return shipped


def _with_error(screen):
    screen._console.append_error("ValueError: a distinctive failure\n")


def test_a_console_that_cannot_be_read_has_nothing_to_report(
        screen, sent, monkeypatch):
    def unreadable():
        raise RuntimeError("the console was torn down")

    monkeypatch.setattr(screen._console, "copy_all", unreadable)
    screen._on_file_issue()
    assert "nothing in the console to report" in screen._console.as_text()
    assert sent == []


def test_an_empty_console_has_nothing_to_report(screen, sent, monkeypatch):
    monkeypatch.setattr(screen._console, "copy_all", lambda: "   \n")
    screen._on_file_issue()
    assert "nothing in the console to report" in screen._console.as_text()
    assert sent == []


def test_a_reporter_that_will_not_import_says_so(screen, sent, monkeypatch):
    import sys

    _with_error(screen)
    monkeypatch.setitem(sys.modules, "spacr.qt.ai.issue_preview", None)
    screen._on_file_issue()
    assert "Issue reporting is unavailable" in screen._console.as_text()
    assert sent == []


def test_closing_the_preview_sends_nothing(screen, sent, monkeypatch):
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    _with_error(screen)
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: QDialog.Rejected)
    screen._on_file_issue()
    assert "The report was not sent." in screen._console.as_text()
    assert sent == []


def test_a_report_that_cannot_be_built_says_why(screen, sent, monkeypatch):
    def unbuildable(*_a, **_k):
        raise ValueError("no fingerprint")

    _with_error(screen)
    monkeypatch.setattr("spacr.qt.ai.issue_report.build_report", unbuildable)
    screen._on_file_issue()
    assert "Could not file the issue: no fingerprint" in \
        screen._console.as_text()
    assert sent == []


def test_a_refused_submission_is_reported_with_its_reason(screen,
                                                         monkeypatch):
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    def refuse(_report):
        raise PermissionError("bad token")

    _with_error(screen)
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: QDialog.Accepted)
    monkeypatch.setattr("spacr.qt.ai.issue_report.submit_report", refuse)
    screen._on_file_issue()
    text = screen._console.as_text()
    assert "Could not file the issue: PermissionError: bad token" in text
    assert "The report was sent" not in text


def test_a_reporter_that_will_not_start_says_so(screen, sent, monkeypatch):
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    _with_error(screen)
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: QDialog.Accepted)
    monkeypatch.setattr(screen._report_jobs, "submit",
                        lambda job, then: False)
    screen._on_file_issue()
    assert "the reporter would not start" in screen._console.as_text()
    assert sent == []


def test_a_pack_applies_every_field_it_can_parse(screen):
    screen._settings.image_size = (120, 120)
    screen._apply_test_data_settings({
        "annotation_column": " infected ",
        "crop_size": "not a number",
        "channels": "1,2,3",
        "image_type": "cell_png",
    })
    assert screen._settings.annotation_column == "infected"
    assert screen._settings.image_size == (120, 120), (
        "an unusable crop size must leave the old one in place")
    assert [str(c) for c in screen._settings.channels] == ["1", "2", "3"]
    assert screen._settings.image_type == "cell_png"


def test_a_pack_with_the_old_spelling_and_no_column_still_applies(screen):
    screen._settings.image_type = "cell_png"
    screen._apply_test_data_settings({"img_size": "96.0",
                                      "image_type": "None"})
    assert screen._settings.annotation_column == "infected"
    assert screen._settings.image_size == (96, 96)
    assert screen._settings.image_type == "cell_png"


def _badge_pixels(qtbot, state):
    from spacr.qt.screens.annotate import _Thumbnail

    tile = _Thumbnail(0)
    qtbot.addWidget(tile)
    tile.resize(120, 120)
    tile.set_badge(state, ("#202020", "#ffffff"))
    image = QImage(120, 120, QImage.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, False)
    tile._paint_badge(painter)
    painter.end()
    box = tile.badge_rect().toRect()
    return [image.pixelColor(x, y).rgba()
            for y in range(box.top(), box.bottom() + 1)
            for x in range(box.left(), box.right() + 1)]


def test_every_badge_is_drawn_and_no_two_look_alike(qtbot):
    confirmed = _badge_pixels(qtbot, "confirmed")
    rejected = _badge_pixels(qtbot, "rejected")
    suggested = _badge_pixels(qtbot, "suggested")
    blank = QColor(0, 0, 0, 0).rgba()
    white = QColor("#ffffff").rgba()
    for drawn in (confirmed, rejected, suggested):
        assert any(p != blank for p in drawn), "the disc was not drawn"
        assert any(p == white for p in drawn), "the glyph was not drawn"
    assert confirmed != rejected
    assert confirmed != suggested
    assert rejected != suggested


def test_no_badge_draws_nothing(qtbot):
    blank = QColor(0, 0, 0, 0).rgba()
    assert set(_badge_pixels(qtbot, None)) == {blank}


def test_a_tile_too_small_for_its_badge_draws_none(qtbot):
    from spacr.qt.screens.annotate import _Thumbnail

    tile = _Thumbnail(0)
    qtbot.addWidget(tile)
    tile.resize(20, 20)
    tile.set_badge("confirmed", ("#202020", "#ffffff"))
    image = QImage(20, 20, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    tile._paint_badge(painter)
    painter.end()
    assert {image.pixelColor(x, y).alpha()
            for x in range(20) for y in range(20)} == {0}
