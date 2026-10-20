"""Behaviour of the model card viewer: cache, loading, failure, scaling, closing.

The network is replaced by byte sources the tests control, the cache by a
temporary folder, and an autouse fixture makes any real connection a failure.
"""
from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import (QBuffer, QByteArray, QCoreApplication, QEvent,
                            QIODevice, Qt)
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from spacr import model_cards as cards
from spacr.qt.widgets import model_card_viewer as mcv
from spacr.qt.widgets.model_card_viewer import ModelCardViewer

KEY = "toxoplasma_pv_v3"
OTHER_KEY = "toxoplasma_pv_v4"


def png_bytes(width=1200, height=900) -> bytes:
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#336699"))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    image.save(buf, "PNG")
    return bytes(data)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    import socket

    def _boom(*_a, **_k):
        raise AssertionError("network access in a card test")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom, raising=False)
    monkeypatch.setattr(cards, "_spacr_home", lambda: tmp_path / "home")


def card(key=KEY):
    return cards._cards()[key]


def make(qtbot, fetch, **kw):
    kw.setdefault("threaded", False)
    w = ModelCardViewer(fetch=fetch, **kw)
    qtbot.addWidget(w)
    w.resize(700, 520)
    return w


def never(url):
    raise AssertionError("the cache should have answered: " + url)


def offline(url):
    raise OSError("offline")


def warm_cache(key=KEY):
    return cards.fetch_pages(card(key)["repo_id"], lambda url: png_bytes())


class Gate:
    """A byte source that waits to be released, so a download stays open."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        self.started.set()
        if not self.release.wait(10):
            raise OSError("gate never released")
        return png_bytes()


def test_a_cached_card_shows_at_once_without_asking_the_network(qtbot):
    warm_cache()
    w = make(qtbot, never, threaded=True)
    w.set_card(card())
    assert w._thread is None
    assert w.showing_image() and w.status_text() == ""
    assert w.page() == 0 and "1 / 3" in w.counter_text()
    assert all(p is not None and not p.isNull() for p in w._pixmaps)
    assert w._image.toolTip() == "Click to enlarge"
    assert w._image.cursor().shape() == Qt.PointingHandCursor


def test_the_link_opens_the_model_on_hugging_face(qtbot):
    w = make(qtbot, offline)
    assert w._link.text() == ""
    w.set_card(card())
    text = w._link.text()
    assert "Open on Hugging Face" in text
    assert f'href="{cards.model_link(card())}"' in text
    assert w._link.openExternalLinks()
    w.set_card(None)
    assert w._link.text() == ""


def test_an_empty_viewer_offers_nothing(qtbot):
    w = make(qtbot, offline)
    assert w.card() is None and w.counter_text() == ""
    assert not w._prev.isEnabled() and not w._next.isEnabled()
    assert not w.showing_image() and w._text.toPlainText() == ""
    w.set_card(card())
    w.set_card(None)
    assert w.card() is None and w.page() == 0 and w.counter_text() == ""
    assert not w._prev.isEnabled() and not w._next.isEnabled()
    assert w._text.toPlainText() == "" and w._link.text() == ""


def test_the_buttons_follow_the_page(qtbot):
    warm_cache()
    w = make(qtbot, never)
    w.set_card(card())
    states = []
    for _ in range(3):
        states.append((w._prev.isEnabled(), w._next.isEnabled(),
                       w.counter_text()))
        w.next_page()
    assert [s[:2] for s in states] == [(False, True), (True, True),
                                       (True, False)]
    assert states[0][2].endswith("1 / 3") and states[2][2].endswith("3 / 3")
    assert states[0][2].startswith("● ○ ○")
    assert states[2][2].startswith("○ ○ ●")
    assert w.page() == 2
    w.go_to(-5)
    assert w.page() == 0
    w.go_to("1")
    assert w.page() == 1


def test_page_changed_fires_once_per_real_change(qtbot):
    warm_cache()
    w = make(qtbot, never)
    w.set_card(card())
    seen = []
    w.page_changed.connect(seen.append)
    w.go_to(0)
    w.previous_page()
    w.next_page()
    w.next_page()
    w.next_page()
    assert seen == [1, 2]


def test_other_keys_are_left_to_qt(qtbot):
    warm_cache()
    w = make(qtbot, never)
    w.set_card(card())
    QTest.keyClick(w, Qt.Key_Right)
    QTest.keyClick(w, Qt.Key_A)
    QTest.keyClick(w, Qt.Key_Up)
    assert w.page() == 1
    QTest.keyClick(w, Qt.Key_Left)
    assert w.page() == 0


def test_the_picture_is_scaled_through_hidpi_to_the_label_size(qtbot,
                                                               monkeypatch):
    warm_cache()
    real = mcv.scaled_for
    calls = []

    def spy(source, target, width, height):
        calls.append((source, target, width, height))
        return real(source, target, width, height)

    monkeypatch.setattr(mcv, "scaled_for", spy)
    w = make(qtbot, never)
    w.show()
    w.set_card(card())
    calls.clear()
    w.resize(500, 400)
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)
    source, target, width, height = calls[-1]
    assert target is w._image and source is w._image._source
    assert (width, height) == (w._image.width(), w._image.height())
    size = w._image.pixmap().deviceIndependentSize()
    assert size.width() <= width + 1 and size.height() <= height + 1


def test_a_label_of_no_size_still_scales_to_one_pixel(qtbot, monkeypatch):
    seen = []

    def spy(source, target, width, height):
        seen.append((width, height))
        return source

    monkeypatch.setattr(mcv, "scaled_for", spy)
    label = mcv._PageLabel()
    qtbot.addWidget(label)
    label.resize(0, 0)
    label.set_source(QPixmap(40, 30))
    assert seen and all(w >= 1 and h >= 1 for w, h in seen)


def test_a_null_pixmap_clears_the_label(qtbot):
    label = mcv._PageLabel()
    qtbot.addWidget(label)
    label.set_source(QPixmap(40, 30))
    assert not label.pixmap().isNull()
    label.set_source(QPixmap())
    assert label.pixmap().isNull()
    label.set_source(QPixmap(40, 30))
    label.set_source(None)
    assert label.pixmap().isNull()


def test_only_a_left_click_on_the_image_asks_to_enlarge(qtbot):
    label = mcv._PageLabel()
    qtbot.addWidget(label)
    clicks = []
    label.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(label, Qt.RightButton)
    QTest.mouseClick(label, Qt.MiddleButton)
    assert clicks == []
    QTest.mouseClick(label, Qt.LeftButton)
    assert clicks == [1]


def test_a_card_that_is_not_enlargeable_says_nothing_and_opens_nothing(qtbot):
    warm_cache()
    w = make(qtbot, never, enlargeable=False)
    w.set_card(card())
    assert w._image.toolTip() == ""
    assert w._image.cursor().shape() == Qt.ArrowCursor
    QTest.mouseClick(w._image, Qt.LeftButton)
    assert w._dialog is None


def test_enlarging_opens_the_same_page_in_a_dialog_from_the_cache(qtbot):
    warm_cache()
    w = make(qtbot, never)
    w.set_card(card())
    w.go_to(2)
    QTest.mouseClick(w._image, Qt.LeftButton)
    dialog = w._dialog
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QDialog) and dialog.isVisible()
    assert dialog.windowTitle() == card()["display_name"]
    assert (dialog.width(), dialog.height()) == (1000, 800)
    big = dialog.findChild(ModelCardViewer)
    assert big.page() == 2 and big.showing_image()
    assert big._image.toolTip() == "" and big._enlargeable is False
    dialog.close()


def test_loading_then_arrival_on_a_worker_thread(qtbot):
    gate = Gate()
    w = make(qtbot, gate, threaded=True)
    w.set_card(card())
    assert gate.started.wait(5)
    assert w.status_text() == "Loading the card images…"
    assert not w.showing_image()
    assert w._text.toPlainText().startswith(card()["display_name"])
    assert "Object F1" in w._text.toPlainText()
    assert not w._prev.isEnabled() and w._next.isEnabled()
    gate.release.set()
    qtbot.waitUntil(w.showing_image, timeout=5000)
    assert w.status_text() == ""
    assert gate.urls[0] == cards.page_url(card()["repo_id"], 1)
    w._thread.join(5)


def test_a_failed_download_shows_the_stored_card_data(qtbot):
    w = make(qtbot, offline)
    w.set_card(card())
    assert w.status_text() == ("The card images could not be downloaded. "
                               "Showing the card data stored with spaCR.")
    assert not w.showing_image()
    text = w._text.toPlainText()
    assert card()["display_name"] in text and "Object F1" in text
    w.next_page()
    assert "Training FOV" in w._text.toPlainText()
    w.next_page()
    assert "parasitophorous" in w._text.toPlainText()


def test_a_failed_download_can_be_retried_by_showing_the_card_again(qtbot):
    state = {"up": False}

    def fetch(url):
        if not state["up"]:
            raise OSError("offline")
        return png_bytes()

    w = make(qtbot, fetch)
    w.set_card(card())
    assert "could not be downloaded" in w.status_text()
    state["up"] = True
    w.set_card(card())
    assert w.showing_image() and w.status_text() == ""


def test_a_download_that_is_not_an_image_is_a_failure_too(qtbot):
    w = make(qtbot, lambda url: b"<html>rate limited</html>")
    w.set_card(card())
    assert "could not be downloaded" in w.status_text()
    assert not w.showing_image()
    assert cards.cached_pages(card()["repo_id"]) == [None, None, None]


def test_a_broken_png_in_the_cache_shows_the_stored_data_not_a_blank(qtbot):
    paths = warm_cache()
    paths[1].write_bytes(cards._PNG_MAGIC + b"broken")
    w = make(qtbot, never)
    w.set_card(card())
    assert w._pixmaps == [None, None, None]
    assert not w.showing_image() and "could not" in w.status_text()


def test_an_answer_for_another_card_is_ignored(qtbot):
    gate = Gate()
    w = make(qtbot, gate, threaded=True)
    w.set_card(card())
    assert gate.started.wait(5)
    w.set_card(card(OTHER_KEY))
    w._on_fetched(card()["repo_id"], OSError("late"))
    assert not w._failed and not w.showing_image()
    gate.release.set()
    qtbot.waitUntil(w.showing_image, timeout=5000)
    w._on_fetched(card()["repo_id"], OSError("later still"))
    assert w.showing_image() and not w._failed
    w._thread.join(5)


def test_closing_during_a_download_does_not_break_the_worker(qtbot):
    gate = Gate()
    errors = []
    previous = threading.excepthook
    threading.excepthook = errors.append
    try:
        w = ModelCardViewer(threaded=True, fetch=gate)
        w.set_card(card())
        assert gate.started.wait(5)
        thread = w._thread
        w.close()
        w.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        gate.release.set()
        thread.join(10)
        assert not thread.is_alive()
        qtbot.wait(50)
    finally:
        threading.excepthook = previous
    assert errors == []


def test_closing_during_a_download_still_completes_the_cache(qtbot):
    gate = Gate()
    w = make(qtbot, gate, threaded=True)
    w.set_card(card())
    assert gate.started.wait(5)
    w.close()
    gate.release.set()
    w._thread.join(10)
    assert all(cards.cached_pages(card()["repo_id"]))


def test_the_enlarged_viewer_falls_back_to_the_stored_data_offline(qtbot):
    w = make(qtbot, offline)
    w.set_card(card())
    w.go_to(1)
    w._enlarge()
    dialog = w._dialog
    qtbot.addWidget(dialog)
    big = dialog.findChild(ModelCardViewer)
    assert big.page() == 1 and not big.showing_image()
    assert "could not be downloaded" in big.status_text()
    dialog.close()
