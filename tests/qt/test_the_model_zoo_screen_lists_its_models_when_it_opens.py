"""The Model Zoo screen fills itself, wraps descriptions and tables scorecards.

Reported 2026-11-10: with spaCR the only heading on, the screen opened with
an empty list, and toggling headings did nothing until the spaCR community
confirmation was answered. Nothing read the catalogue when the screen
opened; only Scan, or a heading that fetches (community, bioimage.io),
started a read, and spaCR only folds rows. Pinned here, as the user sees it:

* opened with spaCR alone on, the list fills with no click, saying it is
  loading while the catalogue is read;
* a heading toggled after that re-folds at once, and an all-folded list
  says why it is empty;
* "trained on" wraps over several lines in a column of sensible width, the
  columns are user-resizable with the last stretching, and dragged widths
  are kept for the next opening;
* a model's scorecard is a table of white text on a navy fill, in both the
  dark and light themes.
"""
from __future__ import annotations

import dataclasses
import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QHeaderView

from spacr import model_zoo as zoo
from spacr.qt.screens import model_zoo as mz
from spacr.qt.screens.model_zoo import ModelZooScreen
from spacr.qt.widgets import model_zoo_picker as mzp

WIDTHS = "model_zoo/screen_column_widths"

LONG = ("Toxoplasma tachyzoite parasitophorous vacuoles: round 6's DsRed and "
        "anti-Toxoplasma-biotin fields plus the Toxoplasma channel of a new "
        "384-well plate. Round 7, 100 epochs, base cpsam_v2, hand-curated "
        "masks reviewed twice before training.")


@pytest.fixture(autouse=True)
def _only_spacr_heading_and_no_stored_widths():
    """Start every test with spaCR alone on and no remembered widths."""
    store = mzp._store()
    saved = {key: store.value(key) for key in (mzp._SOURCES_SETTING, WIDTHS)
             if store.contains(key)}
    store.setValue(mzp._SOURCES_SETTING, "spaCR")
    store.remove(WIDTHS)
    yield
    for key in (mzp._SOURCES_SETTING, WIDTHS):
        store.remove(key)
        if key in saved:
            store.setValue(key, saved[key])


def _entries(tmp_path):
    """Two spaCR rows, one with a long description, and one stock row."""
    out = []
    for key, trained in (("zoo_open_test_pv_v1", LONG),
                         ("zoo_open_test_nuclei_v1", "nuclei, 20 fields")):
        path = tmp_path / f"{key}.pth"
        path.write_bytes(b"x")
        out.append(zoo.ModelEntry(key=key, name=key, path=str(path),
                                  kind="cellpose", source="local",
                                  trained_on=trained, size_bytes=1))
    out.append(zoo.ModelEntry(key="cpsam", name="cpsam", kind="cellpose",
                              source="stock"))
    return out


def _curated(entry):
    """``entry`` carrying a six-row matched-stock scorecard."""
    scores = {"f1": (0.761, 0.377), "boundary_f1_2px": (0.805, 0.223),
              "precision": (0.769, 0.265), "recall": (0.754, 0.656),
              "aji": (0.582, 0.040), "dice": (0.835, 0.091)}
    metrics = {key: {"finetuned": a, "vanilla": b, "delta": a - b}
               for key, (a, b) in scores.items()}
    metrics.update({"holdout": "zoo_open_test_holdout", "n_fields": 3,
                    "n_objects": 40,
                    "scorecard_provenance": {"boundary_tolerance_px": 2,
                                             "stock_model": "cpsam_v2"}})
    return dataclasses.replace(entry, metrics=metrics, _curated_card=True)


def _shown(screen):
    """How many rows of the model list are on screen."""
    table = screen._table
    return sum(1 for r in range(table.rowCount()) if not table.isRowHidden(r))


def test_opened_with_only_spacr_on_the_list_fills_with_no_click(
        qtbot, tmp_path, monkeypatch):
    entries = _entries(tmp_path)
    assert zoo.source_of(entries[0]) == "spaCR"
    assert zoo.source_of(entries[2]) != "spaCR"
    release = threading.Event()

    def slow_catalogue(*_a, **_k):
        release.wait(10)
        return list(entries)

    monkeypatch.setattr(zoo, "catalogue", slow_catalogue)
    screen = ModelZooScreen(threaded=True)
    qtbot.addWidget(screen)
    assert screen.sources.enabled() == ("spaCR",)
    screen.show()
    try:
        qtbot.waitUntil(screen.is_busy, timeout=5000)
        assert screen._list_note_text() == "Loading models…"
        assert screen._table.rowCount() == 0
    finally:
        release.set()
    qtbot.waitUntil(lambda: _shown(screen) == 2, timeout=10000)
    assert screen._list_note_text() == ""
    assert sorted(screen.chosen_entry(g).key for g in range(3)
                  if not screen._table.isRowHidden(screen._row_of_group(g))) \
        == ["zoo_open_test_nuclei_v1", "zoo_open_test_pv_v1"]

    screen.sources.set_on("spaCR", False)
    assert _shown(screen) == 0
    assert screen._list_note_text().startswith("No models under the headings")
    screen.sources.set_on("spaCR", True)
    assert _shown(screen) == 2 and screen._list_note_text() == ""
    screen.close()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)


def test_a_listing_handed_in_before_the_first_show_is_not_reread(
        qtbot, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(zoo, "catalogue",
                        lambda *a, **k: calls.append(1) or [])
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_entries(_entries(tmp_path))
    screen.show()
    qtbot.wait(50)
    assert calls == [] and _shown(screen) == 2


def test_a_first_show_during_a_job_reads_the_catalogue_after_it(
        qtbot, tmp_path, monkeypatch):
    entries = _entries(tmp_path)
    monkeypatch.setattr(zoo, "catalogue", lambda *a, **k: list(entries))
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._busy = True
    screen._load_on_first_show()
    assert screen._first_load_wanted and screen._table.rowCount() == 0
    screen._on_job_settled(True)
    assert _shown(screen) == 2


def test_descriptions_wrap_and_columns_resize_and_are_remembered(
        qtbot, tmp_path):
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    screen.set_entries(_entries(tmp_path))
    table = screen._table
    header = table.horizontalHeader()
    assert table.wordWrap()
    assert header.stretchLastSection()
    for column in range(table.columnCount()):
        assert header.sectionResizeMode(column) == QHeaderView.Interactive
    widths = screen._column_widths()
    assert widths[2] >= mz._px(mz._DEFAULT_WIDTHS[2])
    long_row = screen._row_of_group(
        next(g for g, (stem, _) in enumerate(screen._groups)
             if stem == "zoo_open_test_pv"))
    short_row = screen._row_of_group(
        next(g for g, (stem, _) in enumerate(screen._groups)
             if stem == "zoo_open_test_nuclei"))
    assert table.rowHeight(long_row) > 1.5 * table.rowHeight(short_row)

    header.resizeSection(2, 333)
    assert mz._WIDTHS_SETTING == WIDTHS
    assert mz._remembered_widths()[2] == 333
    screen.set_entries(_entries(tmp_path))
    assert screen._column_widths()[2] == 333

    again = ModelZooScreen(threaded=False)
    qtbot.addWidget(again)
    again.set_entries(_entries(tmp_path))
    assert again._column_widths()[2] == 333


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_scorecard_is_a_table_of_white_text(qtbot, tmp_path, theme,
                                                 monkeypatch):
    from spacr.qt import preferences as prefs

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: theme)
    entries = _entries(tmp_path)
    entries[0] = _curated(entries[0])
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    screen.set_entries(entries)
    group = next(g for g, (stem, _) in enumerate(screen._groups)
                 if stem == "zoo_open_test_pv")
    screen.select(screen._row_of_group(group))
    rows = screen._scorecard_table()
    assert [r[0] for r in rows][:2] == ["F1 @ IoU 0.5", "Boundary F1 (2 px)"]
    assert rows[0][1:] == ["0.761", "0.377", "+0.384"]
    assert len(rows) == 6 and not screen._card_pane.isHidden()
    table = screen._scorecard
    for r in range(table.rowCount()):
        for c in range(table.columnCount()):
            assert table.item(r, c).foreground().color().name() == "#ffffff"
    sheet = table.styleSheet()
    assert mz.SCORECARD_COLOURS["background"] in sheet
    assert "color: #ffffff" in sheet
    image = table.grab().toImage()
    middle = image.pixelColor(image.width() // 2, image.height() - 6)
    assert middle.lightness() < 90

    other = next(g for g, (stem, _) in enumerate(screen._groups)
                 if stem == "zoo_open_test_nuclei")
    screen.select(screen._row_of_group(other))
    assert screen._scorecard_table() == []


def test_scorecard_rows_read_a_shipped_card_when_there_is_no_curated_one():
    card = {"scores": {"rows": [
        {"label": "Dice", "this": 0.8, "stock": 0.5},
        {"label": "Recall", "this": 0.7, "stock": None}]}}
    entry = zoo.ModelEntry(key="x", name="x", kind="cellpose", source="local")
    assert mz._scorecard_rows(entry, card) == [
        ("Dice", "0.800", "0.500", "+0.300"), ("Recall", "0.700", "", "")]
    assert mz._scorecard_rows(entry, {}) == []
    assert mz._score_text("n/a") == "n/a" and mz._score_text("") == ""
