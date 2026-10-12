"""The 3-page model card viewer of the Model Zoo, offline.

The downloader is replaced by a byte source, the cache by a temporary folder,
and an autouse fixture makes any real connection a failure.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest

from spacr import model_cards as cards
from spacr import model_zoo as zoo
from spacr.qt.screens.model_zoo import ModelZooScreen
from spacr.qt.widgets.model_card_viewer import ModelCardViewer

KEYS = {
    "toxoplasma_pv_v1": "toxoplasma-pv-segmentation-cpsam",
    "toxoplasma_pv_v2": "toxoplasma-pv-segmentation-cpsam-r5",
    "toxoplasma_pv_v3": "toxoplasma-pv-segmentation-cpsam-r6",
    "toxoplasma_pv_v4": "toxoplasma-pv-segmentation-cpsam-r7",
    "toxoplasma_pv_v5": "toxoplasma-pv-segmentation-cpsam-r8",
    "toxoplasma_plaque_v1": "toxoplasma-plaque-segmentation-cpsam",
    "toxoplasma_plaque_v2": "toxoplasma-plaque-segmentation-cpsam-r5",
    "toxoplasma_plaque_v3": "toxoplasma-plaque-segmentation-cpsam-r5-geldoc",
    "toxoplasma_plaque_v4": "toxoplasma-plaque-segmentation-cpsam-r6",
}


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


@pytest.fixture
def good_fetch():
    seen = []

    def fetch(url):
        seen.append(url)
        return png_bytes()

    fetch.seen = seen
    return fetch


def offline(url):
    raise OSError("offline")


def a_card(key="toxoplasma_pv_v3"):
    return cards._cards()[key]


def viewer(qtbot, fetch):
    w = ModelCardViewer(threaded=False, fetch=fetch)
    qtbot.addWidget(w)
    w.resize(700, 520)
    return w


def test_every_model_key_maps_to_its_repo():
    assert set(cards.card_keys()) == set(KEYS)
    for key, repo in KEYS.items():
        assert cards.repo_of_key(key) == "einarolafsson/" + repo
        assert cards.page_url("einarolafsson/" + repo, 2).endswith(
            repo + "/resolve/main/cards/page2.png")
    assert cards.repo_of_key("nope") == ""
    assert cards.repo_of_key(None) == ""


def test_the_repo_agrees_with_the_zoo_uri_of_each_model():
    by_key = {e["key"]: e for e in zoo.BUNDLED_REMOTE_MODELS}
    checked = 0
    for key in KEYS:
        found = zoo.hf_repo_of(by_key[key].get("uri"))
        # the early weights live in the shared models dataset or have no URI
        # of their own; every model with a repo is fetched from its card repo
        if found and found != zoo.HF_MODELS_REPO:
            assert found == cards.repo_of_key(key)
            checked += 1
    assert checked >= 5


def test_card_for_entry_by_key_by_uri_and_none():
    class E:
        key = "toxoplasma_pv_v3"
        uri = ""

    assert cards.card_for_entry(E())["repo_id"].endswith("cpsam-r6")

    class ByUri:
        key = "renamed"
        uri = ("https://huggingface.co/einarolafsson/"
               "toxoplasma-pv-segmentation-cpsam-r7/resolve/main/w")

    assert cards.card_for_entry(ByUri())["key"] == "toxoplasma_pv_v4"

    class Other:
        key = "cellpose_cyto"
        uri = ""

    assert cards.card_for_entry(Other()) is None


def test_fallback_text_from_the_card_data():
    card = a_card()
    rows = cards.score_rows(card)
    assert rows[0][0].startswith("Object F1") and rows[0][1].startswith("0.750")
    assert cards.fact_rows(card)
    assert cards.about_lines(card)
    assert "<table" in cards.page_html(card, 1)
    assert "<table" in cards.page_html(card, 2)
    assert cards.page_html(card, 3).startswith("<p>")
    assert cards.model_link(card).startswith("https://huggingface.co/")
    assert cards._num("n/a") == "n/a"
    assert cards.model_link({"repo_id": "a/b"}) == "https://huggingface.co/a/b"
    leak = {"about": {"leak_note": "careful"}}
    assert cards.about_lines(leak) == ["careful"]


def test_fetch_pages_caches_and_rejects_non_png(good_fetch):
    repo = "einarolafsson/x"
    paths = cards.fetch_pages(repo, good_fetch)
    assert len(paths) == 3 and all(p.is_file() for p in paths)
    assert all(cards.cached_pages(repo))
    cards.fetch_pages(repo, good_fetch)
    assert len(good_fetch.seen) == 3
    cards.clear_cache(repo)
    assert not any(cards.cached_pages(repo))
    with pytest.raises(OSError):
        cards.fetch_pages(repo, lambda url: b"<html>")
    assert not any(cards.cached_pages(repo))


def test_default_fetch_uses_urlopen(monkeypatch):
    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"bytes"

    monkeypatch.setattr(cards.urllib.request, "urlopen",
                        lambda url, timeout: R())
    assert cards._fetch("https://x") == b"bytes"


def test_viewer_pages_buttons_keys_and_counter(qtbot, good_fetch):
    w = viewer(qtbot, good_fetch)
    assert w.counter_text() == "" and w.card() is None
    w.set_card(a_card())
    assert w.showing_image() and w.page() == 0
    assert "1 / 3" in w.counter_text()
    assert not w._prev.isEnabled() and w._next.isEnabled()
    pages = []
    w.page_changed.connect(pages.append)
    w._next.click()
    QTest.keyClick(w, Qt.Key_Right)
    QTest.keyClick(w, Qt.Key_Right)
    assert w.page() == 2 and "3 / 3" in w.counter_text()
    assert pages == [1, 2]
    assert not w._next.isEnabled()
    QTest.keyClick(w, Qt.Key_Left)
    QTest.keyClick(w, Qt.Key_Down)
    w._prev.click()
    w.previous_page()
    assert w.page() == 0
    w.go_to(99)
    assert w.page() == 2
    assert "huggingface.co/einarolafsson" in w._link.text()
    assert w.status_text() == ""


def test_the_image_keeps_its_aspect_ratio(qtbot, good_fetch):
    w = viewer(qtbot, good_fetch)
    w.show()
    w.set_card(a_card())
    w.resize(900, 700)
    pm = w._image.pixmap()
    assert pm is not None and not pm.isNull()
    ratio = pm.deviceIndependentSize().width() / pm.deviceIndependentSize().height()
    assert ratio == pytest.approx(4 / 3, rel=0.02)
    w.set_card(None)
    assert w._image.pixmap().isNull() and w.counter_text() == ""


def test_offline_shows_the_card_data(qtbot):
    w = viewer(qtbot, offline)
    w.set_card(a_card())
    assert not w.showing_image()
    assert "could not be downloaded" in w.status_text()
    assert "Object F1" in w._text.toPlainText()
    w.next_page()
    assert "Training FOV" in w._text.toPlainText()
    w.next_page()
    assert "parasitophorous" in w._text.toPlainText()


def test_a_corrupt_cached_image_falls_back(qtbot, good_fetch):
    cards.fetch_pages("einarolafsson/toxoplasma-pv-segmentation-cpsam-r6",
                      good_fetch)
    folder = cards.cache_dir("einarolafsson/toxoplasma-pv-segmentation-cpsam-r6")
    (folder / "page2.png").write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    w = viewer(qtbot, offline)
    w.set_card(a_card())
    assert not w.showing_image() and "could not" in w.status_text()


def test_threaded_fetch_arrives_on_the_gui_thread(qtbot, good_fetch):
    w = ModelCardViewer(threaded=True, fetch=good_fetch)
    qtbot.addWidget(w)
    w.set_card(a_card())
    assert "Loading" in w.status_text()
    qtbot.waitUntil(w.showing_image, timeout=5000)
    # a late answer for a card no longer shown is ignored
    w._on_fetched("einarolafsson/other", RuntimeError("x"))
    assert w.showing_image()
    w.set_card(None)
    w._on_fetched("einarolafsson/other", RuntimeError("x"))


def test_clicking_the_image_opens_a_larger_dialog(qtbot, good_fetch):
    w = viewer(qtbot, good_fetch)
    w.set_card(a_card())
    w.go_to(1)
    w._image.clicked.emit()
    dialog = w._dialog
    qtbot.addWidget(dialog)
    big = dialog.findChild(ModelCardViewer)
    assert big.page() == 1 and big.card() is a_card()
    big._image.clicked.emit()
    assert big._dialog is None
    QTest.mouseClick(w._image, Qt.LeftButton)
    QTest.mouseClick(w._image, Qt.RightButton)
    empty = viewer(qtbot, good_fetch)
    empty._enlarge()
    assert empty._dialog is None


def entry(key, uri=""):
    return zoo.ModelEntry(key=key, name=key, kind="cellpose", source="remote",
                          uri=uri)


def test_zoo_screen_swaps_the_text_card_for_the_viewer(qtbot, monkeypatch,
                                                       good_fetch):
    monkeypatch.setattr(cards, "_fetch", good_fetch)
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_entries([entry("toxoplasma_pv_v3"), entry("cellpose_cyto")])
    rows = {screen.chosen_entry(r).key: r for r in range(2)}
    screen.select(rows["toxoplasma_pv_v3"])
    assert not screen._card_viewer.isHidden() and screen._detail.isHidden()
    assert screen._card_viewer.card()["key"] == "toxoplasma_pv_v3"
    screen.select(rows["toxoplasma_pv_v3"])
    screen.select(rows["cellpose_cyto"])
    assert screen._card_viewer.isHidden() and not screen._detail.isHidden()
    screen.select()
    assert not screen._detail.isHidden()
    screen.select(0, 1)
    assert screen._card_viewer.isHidden()


def test_the_final_epoch_is_recorded_for_pv_v3_and_v4():
    by_key = {e["key"]: e for e in zoo.BUNDLED_REMOTE_MODELS}
    for key in ("toxoplasma_pv_v3", "toxoplasma_pv_v4"):
        epoch = by_key[key]["metrics"]["best_epoch"]
        assert "final epoch 100" in epoch and "2026-10-12" in epoch
    notes = " ".join(by_key["toxoplasma_pv_v3"]["notes"])
    assert "chosen on a held-out" not in notes
    assert "FINAL epoch 100" in notes


def test_picking_another_version_changes_the_card(qtbot, monkeypatch,
                                                  good_fetch):
    monkeypatch.setattr(cards, "_fetch", good_fetch)
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    entries = [zoo._entry_from_mapping(m) for m in zoo.BUNDLED_REMOTE_MODELS]
    screen.set_entries(entries)
    group = next(i for i, (_, pairs) in enumerate(screen._groups)
                 if any(e.key == "toxoplasma_pv_v3" for _, e in pairs))
    pairs = screen._groups[group][1]
    screen.select(screen._row_of_group(group))
    for want in ("toxoplasma_pv_v3", "toxoplasma_pv_v4"):
        screen._version_picked(
            group, next(i for i, (_, e) in enumerate(pairs) if e.key == want))
        assert screen._card_viewer.card()["key"] == want
