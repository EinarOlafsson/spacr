"""Item 468: Plaque Assay has a Plaque mode and a Figure mode, and a switch.

2026-09-21, the maintainer: "you seem to have copied all the settings from
the mask generation live preview and i cannot cheoose the yolo module ...
two modes, one (plaque mode) for the cropped images of plaques ... the other
(figure mode) mode should allow the user to preview the finding of plaques
(the yolo model), the detection of text on the pannel and plaque annotation
with that text, and the mask generation for the plaques. the user should
easily be able to switch with a switcher between Plaque and Figure mode."

Every heavy piece -- the detector, the text reader, Cellpose -- is faked
here; the real run is recorded in the item file.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr import plaque_papers as pp  # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import test_all_module_smoke as smoke  # noqa: E402


def _png(path: Path, size=(400, 400)) -> Path:
    from PIL import Image

    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, size + (3,), dtype=np.uint8)).save(path)
    return path


class _Box:
    def __init__(self, x0, y0, x1, y1, confidence=0.9):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.confidence = confidence


def _detect(image, weights, confidence=0.25, imgsz=640, min_axis_ratio=0.0):
    return [_Box(100, 100, 180, 180), _Box(220, 100, 300, 180)]


def _read_text(path):
    return [pp.Word("A", 60, 60, 75, 75), pp.Word("WT", 120, 80, 160, 95),
            pp.Word("KO", 240, 80, 280, 95)]


def _two_blobs(shape):
    labels = np.zeros(shape[:2], dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[30:45, 30:45] = 2
    return labels


def _segment_path(path):
    return _two_blobs((400, 400))


def _segment_crop(crop):
    return _two_blobs(crop.shape)


@pytest.fixture
def no_papers_check(monkeypatch):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])


@pytest.fixture
def panel(qtbot, no_papers_check):
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def screen(qtbot):
    from spacr.qt.app import MainWindow

    built = MainWindow._build_screen(smoke._FactoryHost(), "analyze_plaques")
    qtbot.addWidget(built)
    return built


def test_the_screen_carries_the_plaque_panel_not_the_mask_one(screen):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    assert isinstance(screen._live_preview, ppv.PlaquePreviewPanel)
    assert not screen._live_preview.findChildren(LivePreviewPanel)


def test_the_switch_is_the_first_thing_on_the_settings_column(screen):
    switch = screen._plaque_mode_switch
    assert screen._settings_layout.itemAt(0).widget() is switch
    assert switch.button("plaque").text() == "Plaque"
    assert switch.button("figure").text() == "Figure"


def test_both_switches_follow_each_other(screen):
    panel = screen._live_preview
    screen._plaque_mode_switch.button("figure").click()
    assert panel.mode() == "figure"
    panel._mode_switch.button("plaque").click()
    assert screen._plaque_mode_switch.mode() == "plaque"


def test_the_other_modes_settings_leave_the_form(screen):
    hidden = set(screen._settings_model.keys_hidden_by_the_run())
    assert set(ppv.FIGURE_ONLY_KEYS) <= hidden
    screen._plaque_mode_switch.button("figure").click()
    hidden = set(screen._settings_model.keys_hidden_by_the_run())
    assert not set(ppv.FIGURE_ONLY_KEYS) & hidden
    assert set(ppv.PLAQUE_ONLY_KEYS) <= hidden


def test_a_mode_written_into_the_form_moves_both_switches(screen):
    field = screen._settings_model._widgets.get(ppv.MODE_KEY)
    if field is None:
        pytest.skip("plaque_mode is not a setting on this tree yet")
    screen._settings_model.set_value_for_key(ppv.MODE_KEY, "figure")
    assert screen._plaque_mode_switch.mode() == "figure"
    assert screen._live_preview.mode() == "figure"


def test_plaque_mode_segments_one_image_and_counts(panel, tmp_path):
    _png(tmp_path / "a.png")
    _png(tmp_path / "b.png")
    panel.load_source_async(str(tmp_path))
    assert panel._picker.count() == 2
    got = []
    panel.preview_ready.connect(got.append)
    assert panel.run_preview(segment=_segment_path)
    assert got and got[0]["count"] == 2
    assert "2 plaques" in panel.preview_status()
    assert not panel._table.isVisible()
    assert panel._controls.isHidden(), "no inline settings row"


def test_the_picker_steps_and_wraps(panel, tmp_path):
    for name in ("a.png", "b.png", "c.png"):
        _png(tmp_path / name)
    panel.load_source_async(str(tmp_path))
    panel._step(-1)
    assert panel.current_path().name == "c.png"
    panel._step(1)
    assert panel.current_path().name == "a.png"


def _figure_panel(panel, tmp_path, confirm=False):
    _png(tmp_path / "fig1.png")
    panel.apply_settings({"plaque_mode": "figure",
                          "confirm_annotations": confirm})
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(detect=_detect, read_text=_read_text,
                             segment=_segment_crop)
    return panel


def test_run_preview_finds_and_reads_but_segments_nothing(panel, tmp_path):
    """2026-09-21: "run preview should detect the plaque wells"."""
    _figure_panel(panel, tmp_path)
    assert panel._table.rowCount() == 2
    first = [panel._table.item(0, c).text() for c in range(8)]
    assert first[1] == "A"
    assert "WT" in first[2]
    assert first[ppv.PLAQUES_COLUMN] == "", "nothing is segmented yet"
    assert panel._row_ok(0), "without confirm_annotations the run measures it"
    assert panel.selected_well() == 0, "the first well is shown on the right"
    assert panel._well_view.has_image()


def test_a_panel_letter_with_no_legend_asks_for_one(panel, tmp_path):
    _figure_panel(panel, tmp_path)
    assert not panel._legend_box.isHidden()
    assert "annotate by hand" in panel._legend_text.text()
    panel._legend_edit.setPlainText(
        "Figure 1. Plaques. (A) WT and KO plaques after 7 days. (B) Areas.")
    panel._use_pasted_legend()
    assert panel._legend_box.isHidden()
    assert pp.read_legends(tmp_path / pp.LEGENDS_FILE)["fig1"].startswith(
        "Figure 1.")
    assert "WT and KO" in panel._table.item(0, 3).text()
    assert "strong" in panel._table.item(0, 5).text()


def test_a_known_legend_is_used_and_nothing_is_asked(panel, tmp_path):
    ppv.write_legend(tmp_path / pp.LEGENDS_FILE, "fig1",
                     "Figure 1. (A) WT and KO plaques.")
    _figure_panel(panel, tmp_path)
    assert panel._legend_box.isHidden()
    assert "WT and KO" in panel._table.item(0, 3).text()


def test_save_writes_what_the_run_reads(panel, tmp_path):
    _figure_panel(panel, tmp_path)
    panel._table.item(1, ppv.CONDITION_COLUMN).setText("knockout")
    panel._table.item(1, ppv.OK_COLUMN).setCheckState(Qt.Unchecked)
    path = panel.save_annotations()
    assert path == tmp_path / pp.ANNOTATIONS_FILE
    saved = pp.read_annotation_overrides(path)
    assert saved[("fig1", 2)] == {"condition": "knockout", "approved": False}
    assert saved[("fig1", 1)]["approved"] is True


def test_a_saved_review_comes_back_on_the_next_preview(panel, tmp_path):
    pp.write_annotation_overrides(tmp_path / pp.ANNOTATIONS_FILE, [
        {"file": "fig1.png", "region": 1, "condition": "parental",
         "approved": True}])
    _figure_panel(panel, tmp_path, confirm=True)
    assert panel._table.item(0, ppv.CONDITION_COLUMN).text() == "parental"
    assert panel._row_ok(0)
    assert not panel._row_ok(1), "confirm on: nobody approved image 2"


def test_confirm_says_only_ok_rows_are_measured(panel, tmp_path):
    _figure_panel(panel, tmp_path, confirm=True)
    assert not panel._confirm_note.isHidden()
    assert "ONLY" in panel._confirm_note.text()
    panel._confirm.setChecked(False)
    assert panel._confirm_note.isHidden()


def test_missing_papers_extra_is_said_with_the_command(qtbot, tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(ppv, "missing_papers_packages",
                        lambda *a, **k: ["ultralytics", "RapidOCR"])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    _png(tmp_path / "fig1.png")
    widget.load_source_async(str(tmp_path))
    widget.set_mode("figure")
    assert not widget._deps_banner.isHidden()
    assert "environment of their own" in widget._deps_text.text()
    assert widget.run_preview() is False
    assert "ultralytics" in widget.preview_status()


def test_install_uses_an_environment_of_its_own(qtbot):
    """Item 469: the figure reader is installed like Cellpose 3, DINOCell and
    SAMCell, into ~/.spacr/backends/papers, never with pip into spaCR."""
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)

    class Dialog:
        installed = True
        ran = False

        def exec(self):
            Dialog.ran = True

    widget._offer_install(dialog=Dialog())
    assert Dialog.ran


def test_an_installed_reader_counts_as_present(monkeypatch):
    import spacr.plaque_papers as pp

    monkeypatch.setattr(pp, "reader_environment", lambda: "/env/papers")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert ppv.missing_papers_packages() == []


def test_missing_packages_are_found_without_importing_them():
    assert ppv.missing_papers_packages(lambda m: m != "ultralytics") == [
        "ultralytics"]
    assert ppv.missing_papers_packages(lambda m: True) == []


def test_a_seeded_model_is_not_written_back_but_tuned_values_are(panel):
    panel.apply_settings({"plaque_model": "toxoplasma_plaque_v2",
                          "diameter": 30})
    panel._diameter.setValue(55)
    out = panel.settings_for_propagation()
    assert out["diameter"] == 55
    assert "plaque_model" not in out
    panel._model_box.setCurrentText("bundled")
    assert panel.settings_for_propagation()["plaque_model"] == "bundled"


def test_use_these_settings_reaches_the_form(screen):
    written = {}
    screen._settings_model.set_value_for_key = (
        lambda k, v: written.__setitem__(k, v))
    screen._live_preview._diameter.setValue(77)
    screen._live_preview.propagate()
    assert written["diameter"] == 77
    assert written["plaque_mode"] == "plaque"


def test_bundled_is_explained(panel):
    panel.apply_settings({"plaque_model": "bundled"})
    assert "historical" in panel._model_note.text()


def test_a_model_that_is_not_here_is_stated_not_fetched(panel, tmp_path,
                                                        monkeypatch):
    from spacr import submodules as sm

    def missing(settings, fetch=True):
        assert fetch is False, "the preview must never download"
        raise sm.ModelZooMissing("not here")

    monkeypatch.setattr(sm, "_resolve_plaque_model", missing)
    _png(tmp_path / "a.png")
    panel.apply_settings({"plaque_model": "toxoplasma_plaque_v2"})
    panel.load_source_async(str(tmp_path))
    panel.run_preview()
    assert "not on this machine" in panel.preview_status()


def test_sizes_and_modes_parse():
    assert ppv.parse_sizes("640,1280") == (640, 1280)
    assert ppv.parse_sizes([960]) == (960,)
    assert ppv.parse_sizes("junk") == ppv.DEFAULT_SIZES
    assert ppv.normalise_mode("FIGURE") == "figure"
    assert ppv.normalise_mode(None) == "plaque"


def test_one_settings_button_opens_one_window_with_two_tabs(qtbot):
    """2026-09-21, the maintainer: "make sure the settings windows follow the
    same format as other settinggggggs windown with rounded edges transparent
    black background, and have one settings button for plaque assay live and
    add several tabs one for figure and one for plaque detection"."""
    from PySide6.QtWidgets import QDialog, QPushButton

    panel = ppv.PlaquePreviewPanel()
    qtbot.addWidget(panel)
    panel.show()
    texts = [b.text() for b in panel.findChildren(QPushButton)
             if b.isVisibleTo(panel)]
    assert texts.count("Settings…") == 1
    assert not any("Figure settings" in t or "Plaque settings" in t
                   for t in texts)
    assert panel._controls.isHidden(), "Plaque mode has no inline row either"

    panel.set_mode("figure")
    panel._settings_btn.click()
    dialog = panel._settings_dialog
    assert isinstance(dialog, QDialog), (
        "a QDialog, so spacr.qt.widgets.glass gives it the rounded "
        "translucent card every settings window has")
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "Figure", "Text detection", "Plaque detection"]
    assert dialog.tabs.isTabVisible(0)
    assert dialog.tabs.currentIndex() == 0
    assert panel._detector_box.isVisibleTo(dialog.figure_tab)
    assert panel._model_box.isVisibleTo(dialog.plaque_tab)
    panel._confidence.setValue(0.4)
    dialog.reject()
    assert panel._settings_dialog is None
    assert panel._detector_row.parent() is panel._controls
    assert panel._confidence.value() == pytest.approx(0.4), "values survive"

    panel.set_mode("plaque")
    dialog = panel.open_settings()
    assert not dialog.tabs.isTabVisible(0), "no Figure tab in Plaque mode"
    assert not dialog.tabs.isTabVisible(1), "no Text detection tab either"
    assert dialog.tabs.currentIndex() == 2
    panel.set_mode("figure")
    assert dialog.tabs.isTabVisible(0), "the window follows the switch"
    assert dialog.tabs.isTabVisible(1)
    dialog.reject()


def test_the_settings_window_gets_the_glass_card(qtbot):
    from spacr.qt.widgets import glass

    panel = ppv.PlaquePreviewPanel()
    qtbot.addWidget(panel)
    dialog = ppv.PlaqueSettingsDialog(panel)
    assert glass.wants_glass(dialog)
    assert glass.glass(dialog)
    dialog.give_back()


def test_clicking_a_box_selects_its_well_and_row(panel, tmp_path):
    """"i should be able to click on any of the plaque wells, or on any of the
    annotated columns in the table ... this should loade that well to the
    right of the figure"."""
    _figure_panel(panel, tmp_path)
    panel._on_figure_clicked(260.0, 140.0)
    assert panel.selected_well() == 1
    assert panel._table.currentRow() == 1 or 1 in {
        i.row() for i in panel._table.selectedItems()}
    assert panel._well_title.text().startswith("Well 2")
    panel._on_figure_clicked(5.0, 5.0)
    assert panel.selected_well() == 1, "a click off every box changes nothing"
    panel._table.selectRow(0)
    assert panel.selected_well() == 0


def test_a_click_is_mapped_through_the_scaled_picture(qtbot):
    view = ppv._ImageView()
    qtbot.addWidget(view)
    view.resize(200, 200)
    view.show()
    view.set_image(np.zeros((400, 800, 3), dtype=np.uint8))
    middle = view.height() / 2.0
    x, y = view.image_point(view.width() / 2.0, middle)
    assert abs(x - 400) < 6 and abs(y - 200) < 6
    assert view.image_point(view.width() / 2.0, 2.0) is None, (
        "above the letterboxed image")


def test_plaque_preview_fills_the_selected_row_and_the_plaques_tab(
        panel, tmp_path):
    """"i should be able to apply the plaque model to that image and find
    all the plaques, which should bthen populate the table row ... another
    table tab should appear so i can see the values for each plaque"."""
    _figure_panel(panel, tmp_path)
    panel.select_well(1)
    assert panel.preview_selected_well(segment=_segment_crop)
    assert panel._table.item(1, ppv.PLAQUES_COLUMN).text() == "2"
    assert panel._table.item(1, ppv.MEAN_AREA_COLUMN).text() == "162"
    assert panel._table.item(0, ppv.PLAQUES_COLUMN).text() == ""
    assert panel._plaque_table.rowCount() == 2
    assert panel._tabs.tabText(1) == "Plaques (2)"
    rows = panel.plaque_table_rows()
    assert {r["well"] for r in rows} == {2}
    assert {r["area_px"] for r in rows} == {100, 225}
    assert all(r["condition"] for r in rows)
    assert rows[0]["solidity"] == pytest.approx(1.0)
    assert "2 plaques" in panel._well_title.text()


def test_find_plaques_in_all_wells_does_every_well(panel, tmp_path):
    _figure_panel(panel, tmp_path)
    assert panel.find_plaques_in_all_wells(segment=_segment_crop)
    assert [panel._table.item(r, ppv.PLAQUES_COLUMN).text()
            for r in range(2)] == ["2", "2"]
    rows = panel.plaque_table_rows()
    assert len(rows) == 4
    assert all(r["area_vs_panel_median"] is not None for r in rows)
    assert "Done: 4 plaques" in panel.preview_status()
    assert not panel._cancel_btn.isEnabled()


def test_cancel_stops_the_queue(panel, tmp_path):
    _figure_panel(panel, tmp_path)
    seen = []

    def segment(crop):
        seen.append(1)
        panel.cancel_preview()
        return _two_blobs(crop.shape)

    panel.find_plaques_in_all_wells(segment=segment)
    assert len(seen) == 1, "the second well was never started"
    assert panel._plaque_table.rowCount() == 0, "a cancelled answer is dropped"


def test_selecting_a_plaque_row_selects_its_well(panel, tmp_path):
    _figure_panel(panel, tmp_path)
    panel.find_plaques_in_all_wells(segment=_segment_crop)
    panel.select_well(0)
    panel._plaque_table.selectRow(3)
    assert panel.selected_well() == 1


def test_plaque_rows_carry_the_runs_per_plaque_values():
    rows = ppv.plaque_rows(_two_blobs((60, 60)))
    assert [r["plaque_id"] for r in rows] == [1, 2]
    assert set(rows[0]) >= {"area_px", "area_vs_well_median", "perimeter_px",
                            "equivalent_diameter_px", "eccentricity",
                            "solidity", "centroid_y", "centroid_x"}
    assert ppv.plaque_rows(np.zeros((5, 5), dtype=int)) == []


def test_region_at_prefers_the_smallest_box():
    big = pp.Region(0, 0, 100, 100)
    small = pp.Region(10, 10, 30, 30)
    assert ppv.region_at([big, small], 20, 20) == 1
    assert ppv.region_at([big, small], 80, 80) == 0
    assert ppv.region_at([big, small], 200, 200) is None


def test_a_paper_becomes_a_folder_the_preview_reads(panel, tmp_path):
    """Item 424, "auto gather the figure ledgend": From a paper... fetches the
    figures and legends.csv into <parent>/<paper>, and the preview loads it."""
    calls = []

    def fetch(reference, dest):
        calls.append((reference, dest))
        dest.mkdir(parents=True)
        _png(dest / "fig1.png")
        ppv.write_legend(dest / pp.LEGENDS_FILE, "fig1",
                         "Figure 1. (A) WT and KO plaques.")
        return {"folder": str(dest), "paper": "10.1371/x", "figures": 1,
                "with_legend": 1, "licence": "CC BY"}

    written = []
    panel.set_propagate_callback(written.append)
    panel.set_mode("figure")
    assert panel._paper_btn.isVisibleTo(panel) or not panel.isVisible()
    assert panel.fetch_paper("doi:10.1371/journal.ppat.1011009", tmp_path,
                             fetch=fetch)
    dest = tmp_path / "10.1371_journal.ppat.1011009"
    assert calls == [("doi:10.1371/journal.ppat.1011009", dest)]
    assert written == [{"src": str(dest)}]
    assert panel.current_path() == dest / "fig1.png"
    assert "1 figures, 1 with legends (licence CC BY)" in \
        panel._paper_note.text()
    assert panel._paper_btn.isEnabled()
    panel.run_preview(detect=_detect, read_text=_read_text)
    assert panel._legend_box.isHidden(), "the fetched legend is used"
    assert "WT and KO" in panel._table.item(0, 3).text()


def test_a_failed_fetch_is_said_and_the_button_comes_back(panel, tmp_path):
    def fetch(reference, dest):
        raise RuntimeError("Europe PMC has no figures for it")

    assert panel.fetch_paper("12345", tmp_path, fetch=fetch)
    assert "Europe PMC has no figures" in panel.preview_status()
    assert panel._paper_btn.isEnabled()


def test_paper_folder_names_are_safe():
    assert ppv.paper_folder_name("https://doi.org/10.1/a b") == "10.1_a_b"
    assert ppv.paper_folder_name("/x/y/My Paper.pdf") == "My_Paper"
    assert ppv.paper_folder_name("PMC123") == "PMC123"
    assert ppv.paper_folder_name("") == "paper"


def test_the_paper_dialog_hands_back_what_was_typed(qtbot):
    dialog = ppv.PaperDialog(None, "/data")
    qtbot.addWidget(dialog)
    dialog.reference.setText("  PMC9744290 ")
    assert dialog.values() == ("PMC9744290", "/data")


def test_the_screen_writes_the_fetched_folder_into_src(screen, tmp_path):
    screen._propagate_live_settings({"src": str(tmp_path)})
    assert screen._settings_model.collect()["src"] == str(tmp_path)


def _read_text_with_a_row_label(path):
    return _read_text(path) + [pp.Word("+ATc", 40, 130, 80, 145)]


def _text_panel(panel, tmp_path):
    _png(tmp_path / "fig1.png")
    panel.apply_settings({"plaque_mode": "figure"})
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(detect=_detect,
                             read_text=_read_text_with_a_row_label)
    return panel


def _condition(panel, row=0):
    return panel._table.item(row, ppv.CONDITION_COLUMN).text()


def test_text_detection_settings_re_propose_the_condition_at_once(
        panel, tmp_path):
    """2026-09-21: "also add a tab for text detection, as this seems to pick
    up the text correctly but it is not annotating correctly allways so
    whatever settings you can add there please do." Every change re-annotates
    the words already read -- the detector and OCR are not asked again."""
    calls = []

    def read(path):
        calls.append(path)
        return _read_text_with_a_row_label(path)

    _png(tmp_path / "fig1.png")
    panel.apply_settings({"plaque_mode": "figure"})
    panel.load_source_async(str(tmp_path))
    panel.run_preview(detect=_detect, read_text=read)
    assert _condition(panel) == "WT / +ATc"

    t = panel._text
    t["text_order"].setCurrentText("left,above")
    assert _condition(panel) == "+ATc / WT"
    t["text_separator"].setText(" | ")
    assert _condition(panel) == "+ATc | WT"
    t["text_use_left"].setChecked(False)
    assert _condition(panel) == "WT"
    t["text_use_left"].setChecked(True)
    t["text_ignore"].setText("^WT$")
    assert _condition(panel) == "+ATc"
    t["text_ignore"].setText("")
    t["text_reach_above"].setValue(0.0)
    assert _condition(panel) == "+ATc", "the header is out of reach"
    assert len(calls) == 1, "tuning never reads the figure again"
    assert "proposed again" in panel.preview_status()


def test_the_panel_letter_reach_decides_the_panel(panel, tmp_path):
    _text_panel(panel, tmp_path)
    assert panel._table.item(0, 1).text() == "A"
    panel._text["text_panel_reach"].setValue(0.0)
    assert panel._table.item(0, 1).text() == ""


def test_a_low_confidence_word_can_be_dropped(panel, tmp_path):
    def read(path):
        return [pp.Word("WT", 120, 80, 160, 95, 0.4),
                pp.Word("KO", 240, 80, 280, 95, 0.9)]

    _png(tmp_path / "fig1.png")
    panel.apply_settings({"plaque_mode": "figure"})
    panel.load_source_async(str(tmp_path))
    panel.run_preview(detect=_detect, read_text=read)
    assert _condition(panel, 0) == "WT"
    panel._text["text_min_confidence"].setValue(0.5)
    assert "WT" not in _condition(panel, 0)
    assert _condition(panel, 1) == "KO"


def test_the_second_reading_asks_for_a_new_run(panel, tmp_path):
    _text_panel(panel, tmp_path)
    before = _condition(panel)
    panel._text["text_reread_scale"].setValue(5)
    assert "Run preview" in panel.preview_status()
    assert _condition(panel) == before


def test_text_settings_are_seeded_and_written_back(panel):
    panel.apply_settings({"plaque_mode": "figure", "text_reach_left": 2.5,
                          "text_use_below": False, "text_ignore": "^\\d+$",
                          "text_order": "left,above,below",
                          "text_separator": " - ", "text_reread_scale": 4})
    assert panel._text["text_reach_left"].value() == pytest.approx(2.5)
    assert not panel._text["text_use_below"].isChecked()
    out = panel.settings_for_propagation()
    assert out["text_reach_left"] == pytest.approx(2.5)
    assert out["text_use_below"] is False
    assert out["text_ignore"] == "^\\d+$"
    assert out["text_order"] == "left,above,below"
    assert out["text_separator"] == " - "
    assert out["text_reread_scale"] == 4
    options = panel.text_options()
    assert options.order == ("left", "above", "below")
    assert options.ignore == ("^\\d+$",)


def test_the_text_tab_holds_a_control_per_setting(qtbot):
    panel = ppv.PlaquePreviewPanel()
    qtbot.addWidget(panel)
    panel.set_mode("figure")
    dialog = panel.open_settings("text")
    assert dialog.tabs.currentIndex() == 1
    for key in ppv.TEXT_KEYS:
        assert panel._text[key].isVisibleTo(dialog.text_tab), key
    assert panel._text["text_reach_above"].toolTip(), "the setting's tooltip"
    dialog.reject()
    assert panel._text["text_order"].parent() is panel._controls


def test_the_form_hides_the_text_settings_in_plaque_mode(screen):
    hidden = set(screen._settings_model.keys_hidden_by_the_run())
    assert set(ppv.TEXT_KEYS) & set(screen._settings_model._widgets) <= hidden
    screen._plaque_mode_switch.button("figure").click()
    hidden = set(screen._settings_model.keys_hidden_by_the_run())
    assert not set(ppv.TEXT_KEYS) & hidden


def test_use_these_settings_writes_the_text_settings(screen):
    from spacr.qt.preview_registry import PREVIEWS

    for key in ppv.TEXT_KEYS:
        assert PREVIEWS["analyze_plaques"].propagation.get(key) == key
    screen._plaque_mode_switch.button("figure").click()
    screen._live_preview._text["text_separator"].setText(" ; ")
    screen._live_preview.propagate()
    assert screen._settings_model.collect()["text_separator"] == " ; "


@pytest.mark.parametrize("mode", ["plaque", "figure"])
def test_no_settings_control_paints_over_the_mode_switch(qtbot, mode):
    """Reported 2026-09-21: "when i start live preview in plaque assay i get a
    field that covers the mode to the left of the plaque and figure button
    ... as soon as i pressed settings it went away". Settings never opened."""
    from PySide6.QtCore import QRect

    panel = ppv.PlaquePreviewPanel()
    qtbot.addWidget(panel)
    panel.resize(1000, 800)
    panel.set_mode(mode)
    panel.show()
    qtbot.waitExposed(panel)
    assert panel._stow_free_widgets() == 0, "nothing left homeless"
    lent = [panel._detector_box, panel._sizes, panel._confidence,
            panel._read_text, panel._confirm, panel._model_row,
            panel._diameter, panel._flow, panel._cellprob,
            *panel._text.values()]
    assert not [w for w in lent if w.isVisible()]
    switch = panel._mode_switch
    area = QRect(switch.mapTo(panel, switch.rect().topLeft()), switch.size())
    area = area.united(QRect(0, 0, switch.x() + 1, switch.height()))
    for child in panel.findChildren(QWidget):
        if not child.isVisible() or child.isWindow():
            continue
        if child is switch or switch.isAncestorOf(child):
            continue
        if child.isAncestorOf(switch):
            continue
        box = QRect(child.mapTo(panel, child.rect().topLeft()), child.size())
        assert not box.intersects(area), (
            f"{child.metaObject().className()} {child.objectName()!r} at "
            f"{box} paints over the mode switch at {area}")


class _FakeZoo:
    """The zoo picker, answered without a window or a download."""

    opened_with = []

    def __init__(self, kinds=None, parent=None):
        _FakeZoo.opened_with.append(tuple(kinds or ()))
        self._kinds = kinds

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.Accepted

    def chosen_path(self):
        return "/models/downloaded.pt"

    def selected_entry(self):
        from types import SimpleNamespace

        key = ("toxoplasma_well_detector_v9" if "detector" in self._kinds
               else "toxoplasma_plaque_v9")
        return SimpleNamespace(key=key)


@pytest.mark.parametrize("button, box, kind, key", [
    ("_detector_zoo_btn", "_detector_box", "detector",
     "toxoplasma_well_detector_v9"),
    ("_model_zoo_btn", "_model_box", "cellpose", "toxoplasma_plaque_v9"),
])
def test_the_detector_and_plaque_model_fields_have_a_model_zoo_button(
        panel, monkeypatch, button, box, kind, key):
    from spacr.qt.widgets import model_zoo_picker

    monkeypatch.setattr(model_zoo_picker, "ModelZooPicker", _FakeZoo)
    _FakeZoo.opened_with.clear()
    getattr(panel, button).click()
    assert _FakeZoo.opened_with == [(kind,)]
    assert getattr(panel, box).currentText() == key


def test_the_settings_window_shows_the_zoo_buttons(panel):
    dialog = ppv.PlaqueSettingsDialog(panel)
    assert panel._detector_zoo_btn.parent() is panel._detector_row
    assert panel._detector_row in dialog._lent()
    assert panel._model_zoo_btn.parent() is panel._model_row
    dialog.give_back() if hasattr(dialog, "give_back") else None


# 2026-09-21, the maintainer: "in plaque mode i can only see the outlines, i
# should be able to right click on the image and get settings and choose to
# see outlines, and pick their color including random color, their
# thickness, or to see the objects as overlayed with fill and i should be
# able to pick the fill color and opacity. there should also be several tabs
# where the first is the image with overlays, the second the objects without
# overlay, then cell probability and a flows tab."


def _flat_png(path: Path, value=40, size=(60, 60)) -> Path:
    from PIL import Image

    Image.fromarray(np.full(size + (3,), value, dtype=np.uint8)).save(path)
    return path


def _square(shape=(60, 60)):
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[35:50, 35:55] = 2
    return labels


def _fake_flows(shape=(60, 60)):
    flow = np.zeros(shape + (3,), dtype=np.uint8)
    flow[..., 0] = 200
    prob = np.full(shape, -6.0, dtype=np.float32)
    prob[10:30, 10:30] = 6.0
    return {"flow_rgb": flow, "cellprob": prob}


def _with_flows(path):
    return _square(), _fake_flows()


@pytest.fixture
def fresh_style(monkeypatch):
    monkeypatch.setitem(ppv._SESSION, "style", ppv.OverlayStyle())


@pytest.fixture
def ran(panel, tmp_path, fresh_style):
    _flat_png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(segment=_with_flows)
    return panel


def _overlay(panel):
    return panel._view.array()


def test_the_image_has_four_tabs_in_order(panel):
    tabs = panel._image_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Overlay", "Objects", "Cell probability", "Flows"]
    assert tabs.currentIndex() == 0
    assert tabs.widget(0) is panel._view


def test_tabs_without_data_say_so_instead_of_staying_blank(
        panel, tmp_path, fresh_style):
    _flat_png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    for view in (panel._objects_view, panel._prob_view, panel._flow_view):
        assert view.text() == "Press Run preview to see this."
    assert panel.run_preview(segment=lambda p: _square())
    assert panel._objects_view.array() is not None
    assert panel._prob_view.text() == "This run gave no cell probability map."
    assert panel._flow_view.text() == "This run gave no flows."


def test_the_tabs_show_the_mask_cellprob_and_flows_of_the_run(ran):
    objects = ran._objects_view.array()
    assert objects.shape == (60, 60, 3)
    assert not objects[0, 0].any(), "background is black, no image"
    assert objects[20, 20].any() and objects[40, 45].any()
    assert tuple(objects[20, 20]) != tuple(objects[40, 45]), (
        "one colour per object")
    prob = ran._prob_view.array()
    assert prob[20, 20].sum() > prob[0, 0].sum(), "high probability is bright"
    assert (ran._flow_view.array() == _fake_flows()["flow_rgb"]).all()


def test_the_default_overlay_is_the_yellow_outline(ran):
    shown = _overlay(ran)
    assert tuple(shown[10, 20]) == ppv.OUTLINE_COLOUR
    assert tuple(shown[20, 20]) == (40, 40, 40), "inside is untouched"


def test_right_click_opens_the_overlay_options(ran, monkeypatch):
    from PySide6.QtCore import QPoint

    shown = []
    monkeypatch.setattr(ppv.PlaquePreviewPanel, "_exec_menu",
                        staticmethod(lambda menu, pos: shown.append(menu)))
    ran._view.context_requested.emit(QPoint(5, 5))
    assert shown
    texts = [a.text() for a in shown[0].actions() if a.text()]
    assert texts == ["Outlines", "Filled overlay",
                     "Random colour per object", "Overlay settings…",
                     "Save picture…"]
    settings = [a for a in shown[0].actions()
                if a.text() == "Overlay settings…"][0]
    settings.trigger()
    dialog = ran._overlay_dialog
    assert isinstance(dialog, ppv.PlaqueOverlayDialog)
    from spacr.qt.widgets import glass

    assert glass.wants_glass(dialog)
    dialog.reject()


def test_a_real_right_click_event_reaches_the_menu(ran, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent

    shown = []
    monkeypatch.setattr(ppv.PlaquePreviewPanel, "_exec_menu",
                        staticmethod(lambda menu, pos: shown.append(pos)))
    event = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(3, 3),
                              QPoint(30, 40))
    ran._objects_view.contextMenuEvent(event)
    assert shown == [QPoint(30, 40)]


def test_outline_colour_and_thickness_change_the_pixels(ran):
    ran.open_overlay_settings()
    dialog = ran._overlay_dialog
    dialog.outline_colour.set_fixed("#ff0000")
    assert tuple(_overlay(ran)[10, 20]) == (255, 0, 0)
    assert tuple(_overlay(ran)[12, 20]) == (40, 40, 40)
    dialog.thickness.setValue(5)
    assert tuple(_overlay(ran)[12, 20]) == (255, 0, 0), "thicker outline"
    assert ran.overlay_style().outline_thickness == 5
    dialog.reject()


def test_random_outline_colour_differs_per_object(ran):
    from dataclasses import replace

    ran.set_overlay_style(replace(ran.overlay_style(),
                                  outline_colour=ppv.RANDOM_COLOUR))
    shown = _overlay(ran)
    first, second = tuple(shown[10, 20]), tuple(shown[35, 45])
    assert first != (40, 40, 40) and second != (40, 40, 40)
    assert first != second
    assert first != ppv.OUTLINE_COLOUR


def test_the_menu_switches_to_fill_and_to_random(ran, monkeypatch):
    menus = []
    monkeypatch.setattr(ppv.PlaquePreviewPanel, "_exec_menu",
                        staticmethod(lambda menu, pos: menus.append(menu)))
    ran._on_view_context(None)
    fill = [a for a in menus[0].actions() if a.text() == "Filled overlay"][0]
    fill.trigger()
    assert ran.overlay_style().display == "fill"
    assert tuple(_overlay(ran)[10, 20]) != ppv.OUTLINE_COLOUR
    assert tuple(_overlay(ran)[20, 20]) != (40, 40, 40), "inside is filled"


def test_fill_colour_and_opacity_change_the_pixels(ran):
    ran.open_overlay_settings()
    dialog = ran._overlay_dialog
    dialog.display.setCurrentIndex(1)
    assert dialog.fill_group.isEnabled() and not dialog.outline_group.isEnabled()
    dialog.fill_colour.set_fixed((0, 0, 255))
    dialog.opacity.setValue(100)
    assert tuple(_overlay(ran)[20, 20]) == (0, 0, 255)
    assert tuple(_overlay(ran)[5, 5]) == (40, 40, 40), "background untouched"
    dialog.opacity.setValue(50)
    assert tuple(_overlay(ran)[20, 20]) == (20, 20, 148)
    assert dialog.opacity_slider.value() == 50
    dialog.opacity.setValue(0)
    assert tuple(_overlay(ran)[20, 20]) == (40, 40, 40)
    dialog.fill_colour.kind.setCurrentIndex(1)
    dialog.opacity.setValue(100)
    shown = _overlay(ran)
    assert tuple(shown[20, 20]) != tuple(shown[40, 45])
    dialog.reject()


def test_the_style_redraws_without_segmenting_again(ran, monkeypatch):
    from dataclasses import replace

    calls = []
    monkeypatch.setattr(ppv, "plaque_pass",
                        lambda *a, **k: calls.append(a) or {})
    monkeypatch.setattr(ppv, "segment_plaque_image",
                        lambda *a, **k: calls.append(a))

    ran.set_overlay_style(replace(ran.overlay_style(), display="fill",
                                  fill_opacity=100,
                                  fill_colour=(0, 255, 0)))
    assert not calls
    assert tuple(_overlay(ran)[20, 20]) == (0, 255, 0)


def test_the_style_lasts_the_session(qtbot, ran, no_papers_check):
    from dataclasses import replace

    ran.set_overlay_style(replace(ran.overlay_style(), outline_thickness=3))
    later = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(later)
    assert later.overlay_style().outline_thickness == 3


def test_figure_mode_draws_wells_in_the_chosen_style(panel, tmp_path,
                                                     fresh_style,
                                                     monkeypatch):
    from dataclasses import replace

    _figure_panel(panel, tmp_path)
    assert not panel._image_tabs.tabBar().isVisibleTo(panel)
    panel.select_well(0)
    assert panel.preview_selected_well(segment=_segment_crop)
    crop = panel._well_view.array()
    assert tuple(crop[10, 15]) == ppv.OUTLINE_COLOUR
    panel.set_overlay_style(replace(panel.overlay_style(), display="fill",
                                    fill_colour=(0, 0, 255),
                                    fill_opacity=100))
    crop = panel._well_view.array()
    assert tuple(crop[15, 15]) == (0, 0, 255)
    overlay = panel._figure["overlay"]
    assert tuple(overlay[100 + 15, 100 + 15]) == (0, 0, 255)
    menus = []
    monkeypatch.setattr(ppv.PlaquePreviewPanel, "_exec_menu",
                        staticmethod(lambda menu, pos: menus.append(menu)))
    panel._well_view.context_requested.emit(None)
    assert menus, "the well picture offers the same options"


def test_a_right_click_does_not_select_a_well(panel, tmp_path):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    _figure_panel(panel, tmp_path)
    got = []
    panel._view.clicked.connect(lambda x, y: got.append((x, y)))
    panel._view.resize(400, 400)
    event = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(200, 200),
                        QPointF(200, 200), Qt.RightButton, Qt.RightButton,
                        Qt.NoModifier)
    panel._view.mousePressEvent(event)
    assert not got


def test_segment_plaque_image_can_hand_back_its_flows():
    from spacr.plaque import segment_plaque_image

    class _Tensor:
        def __init__(self, array):
            self.array = array

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class _Model:
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D,
                                     stitch_threshold=stitch_threshold)
            flows = _fake_flows()
            return (_Tensor(_square()),
                    [_Tensor(flows["flow_rgb"]), None,
                     _Tensor(flows["cellprob"])], None)

    image = np.zeros((60, 60, 3), dtype=np.uint8)
    labels = segment_plaque_image(_Model(), image, {})
    assert isinstance(labels, np.ndarray) and labels.max() == 2
    labels, flows = segment_plaque_image(_Model(), image, {},
                                         return_flows=True)
    assert flows["flow_rgb"].shape == (60, 60, 3)
    assert flows["cellprob"].dtype == np.float32
    assert flows["cellprob"][20, 20] == pytest.approx(6.0)


def test_a_result_without_flows_gives_none():
    from spacr.plaque import plaque_flow_outputs

    assert plaque_flow_outputs((np.zeros((4, 4)),)) == {
        "flow_rgb": None, "cellprob": None}
