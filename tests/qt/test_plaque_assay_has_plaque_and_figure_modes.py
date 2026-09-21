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

from spacr import plaque_papers as pp  # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402

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
    assert panel._figure_row.isHidden()


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


def test_figure_and_plaque_settings_have_a_button_each(qtbot):
    """2026-09-21, the maintainer: "the settings for the figure should be in
    figure settings, the settings for the plaque detections in plaque
    settings"."""
    panel = ppv.PlaquePreviewPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_mode("figure")
    assert panel._figure_settings_btn.isVisibleTo(panel)
    assert panel._plaque_settings_btn.isVisibleTo(panel)
    assert panel._figure_row.parent() is panel._figure_popup
    assert panel._plaque_controls.parent() is panel._plaque_popup
    panel._figure_settings_btn.click()
    assert panel._figure_popup.isVisible()
    assert panel._detector_box.isVisibleTo(panel._figure_popup)
    panel._figure_popup.hide()
    panel._plaque_settings_btn.click()
    assert panel._plaque_popup.isVisible()
    assert panel._model_box.isVisibleTo(panel._plaque_popup)
    assert not panel._detector_box.isVisibleTo(panel._plaque_popup)
    panel._plaque_popup.hide()
    panel.set_mode("plaque")
    assert not panel._figure_settings_btn.isVisibleTo(panel)
    assert panel._plaque_controls.parent() is panel._controls
    assert panel._plaque_controls.isVisibleTo(panel)
    assert not panel._figure_row.isVisibleTo(panel)
    panel.set_mode("figure")
    assert panel._plaque_controls.parent() is panel._plaque_popup


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
