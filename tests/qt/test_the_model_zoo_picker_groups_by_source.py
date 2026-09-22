"""The model zoo is five clickable sources, not one list and a boolean.

Item 440. The picker showed every catalogue row in one list with a single
"show unvetted community uploads" checkbox. Ten bundled rows, the stock
Cellpose-SAM weights, bioimage.io's collection, the Cellpose 3 backend's own
models and a shared catalogue is more than a list can carry, and a boolean
that means "also show these" cannot fold away the four groups a user is not
looking at.

Two properties carry the feature, and each is a way it could mislead:

  * **the partition is total and disjoint** -- every row the catalogue offers
    lands under exactly one heading, decided from the row itself. A row that
    lands nowhere is a model the user cannot see, which is worse than one
    under the wrong heading;
  * **the warning survives the redesign**. The checkbox said out loud what an
    unvetted upload is before showing one. A heading that silently listed them
    would be a regression dressed as a redesign.
"""
from __future__ import annotations

import pytest

from spacr import model_zoo as mz
from spacr.qt.widgets import model_zoo_picker as mzp


@pytest.fixture(autouse=True)
def _the_heading_preference_is_put_back(qapp):
    """Restore the persisted headings after every test in this file.

    The strip writes through QSettings on every click, which is the feature.
    It is also process-wide ambient state, so a test that leaves "spaCR" off
    opens every later picker with the models this project ships folded away
    -- measured: six tests in ``test_model_zoo_picker`` failed that way, all
    of them reading as "the picker lost its selection".
    """
    from PySide6.QtCore import QSettings

    key = mzp._SOURCES_SETTING
    settings = QSettings()
    had = settings.contains(key)
    before = settings.value(key, "") if had else None
    yield
    settings = QSettings()
    if had:
        settings.setValue(key, before)
    else:
        settings.remove(key)


def _catalogue_rows():
    """Every row the picker can offer, without touching the network."""
    rows = list(mz.catalogue(remote=True, block=False))
    rows += list(mz.community_entries())
    return rows


def test_the_five_sources_are_named_and_two_of_them_are_on_by_default():
    """The maintainer named the five and said which two start on."""
    assert mz.ZOO_SOURCES == ("cellposeSAM", "spaCR", "spaCR community",
                              "bioimage.io", "cellpose3")
    assert mz.DEFAULT_ZOO_SOURCES == ("cellposeSAM", "spaCR")


def test_every_catalogue_row_lands_in_exactly_one_source():
    """Total and disjoint, decided from the row rather than from a list.

    The partition is the whole feature: a row that matches no rule would
    vanish from a picker whose headings are the only way in.
    """
    rows = _catalogue_rows()
    assert rows, "the catalogue offered nothing to partition"
    seen = {}
    for entry in rows:
        source = mz.source_of(entry)
        assert source in mz.ZOO_SOURCES, (entry.key, entry.source, source)
        seen.setdefault(source, []).append(id(entry))
    landed = sum(len(ids) for ids in seen.values())
    assert landed == len(rows)
    flat = [i for ids in seen.values() for i in ids]
    assert len(set(flat)) == len(flat), "a row landed under two headings"


def test_the_bundled_spacr_models_are_under_spacr():
    """The ten rows the maintainer trained, including cell_from_hoechst."""
    by_key = {e.key: e for e in _catalogue_rows()}
    for key in ("toxoplasma_pv_v1", "toxoplasma_pv_v2", "toxoplasma_pv_v3",
                "toxoplasma_plaque_v1", "toxoplasma_well_detector_v1",
                "toxoplasma_well_detector_v2", "toxoplasma_from_cellmask_v1",
                "toxoplasma_from_hoechst_v1", "nuclei_from_cellmask_v1",
                "cell_from_hoechst_v1"):
        assert key in by_key, f"{key} is not in the catalogue at all"
        assert mz.source_of(by_key[key]) == "spaCR", key


def test_the_cellpose_three_models_are_under_cellpose3():
    """cyto, cyto2, cyto3 and nuclei belong to the Cellpose 3 backend."""
    rows = {e.name: e for e in mz._cellpose3_model_entries()}
    assert set(rows) >= {"cyto", "cyto2", "cyto3", "nuclei"}
    for entry in rows.values():
        assert mz.source_of(entry) == "cellpose3", entry.name


def test_the_stock_cellpose_sam_weights_are_under_cellposesam():
    """The picker's own stock row, and the catalogue's."""
    assert mz.source_of(mzp.ModelZooPicker.STOCK_MODEL) == "cellposeSAM"
    for entry in mz.stock_cellpose_entries():
        assert mz.source_of(entry) == "cellposeSAM", entry.name


def test_an_unrecognised_row_goes_under_spacr_and_says_so(caplog):
    """A model a user cannot see is worse than one filed wrongly."""
    from types import SimpleNamespace

    stranger = SimpleNamespace(key="mystery_v1", name="mystery", kind="cellpose",
                               source="from-the-future", uri="", path="")
    with caplog.at_level("INFO", logger="spacr.model_zoo"):
        assert mz.source_of(stranger) == "spaCR"
    assert any("mystery" in record.getMessage() for record in caplog.records)


@pytest.fixture
def picker(qapp, tmp_path, monkeypatch):
    """A picker whose downloads would land in a throwaway folder."""
    monkeypatch.setattr(mzp.ModelZooPicker, "_warm_the_community_catalogue",
                        lambda self: None)
    monkeypatch.setattr(mzp, "DEFAULT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(tmp_path))
    dialog = mzp.ModelZooPicker()
    yield dialog
    dialog.reject()
    dialog.deleteLater()


def _shown_sources(picker):
    """Which sources the rows currently on screen come from."""
    out = set()
    for row, (_stem, pairs) in enumerate(picker._groups):
        if picker.table.isRowHidden(row):
            continue
        out.add(mz.source_of(pairs[picker._chosen[_stem]][1]))
    return out


def test_the_old_boolean_is_gone(picker):
    """"spaCR community" replaces it; two controls for one idea is one too
    many."""
    assert not hasattr(picker, "community_toggle")


def test_opening_it_shows_two_blue_headings_and_only_their_rows(picker):
    """cellposeSAM and spaCR, and nothing else."""
    strip = picker.sources
    assert set(strip.enabled()) == {"cellposeSAM", "spaCR"}
    for name in mz.ZOO_SOURCES:
        assert strip.is_on(name) == (name in mz.DEFAULT_ZOO_SOURCES), name
    assert _shown_sources(picker) <= {"cellposeSAM", "spaCR"}


def test_a_heading_that_is_on_is_blue_and_one_that_is_off_is_muted(picker):
    """The colour IS the state -- there is no other mark on the heading."""
    from spacr.qt.theme import active_palette

    palette = active_palette()
    on = picker.sources.heading("spaCR")
    off = picker.sources.heading("cellpose3")
    assert palette["accent"].lower() in on.styleSheet().lower()
    assert palette["fg_muted"].lower() in off.styleSheet().lower()


def test_clicking_a_heading_folds_its_rows_in_and_out(picker):
    """cellpose3's rows exist all along; the heading decides whether they
    are on screen."""
    strip = picker.sources
    assert "cellpose3" not in _shown_sources(picker)
    strip.set_on("cellpose3", True)
    assert "cellpose3" in _shown_sources(picker)
    strip.set_on("cellpose3", False)
    assert "cellpose3" not in _shown_sources(picker)


def test_turning_on_the_community_heading_warns_once(picker, monkeypatch):
    """Exactly as the checkbox did, and not a second time."""
    from PySide6.QtWidgets import QMessageBox

    asked = []

    def _warning(parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return QMessageBox.Yes

    monkeypatch.setattr(mzp.QMessageBox, "warning", staticmethod(_warning))
    picker.sources.set_on("spaCR community", True)
    assert len(asked) == 1
    title, text = asked[0]
    assert "not vetted" in title.lower()
    assert "nobody has checked what they are" in text.lower()
    picker.sources.set_on("spaCR community", False)
    picker.sources.set_on("spaCR community", True)
    assert len(asked) == 1, "it asked again after it had been answered"


def test_declining_the_warning_leaves_the_heading_off(picker, monkeypatch):
    """Cancel means cancel: the rows stay away and the heading stays muted."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        mzp.QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.Cancel))
    picker.sources.set_on("spaCR community", True)
    assert not picker.sources.is_on("spaCR community")
    assert "spaCR community" not in _shown_sources(picker)


def test_the_model_zoo_page_folds_by_the_same_headings(qapp, qtbot):
    """Three places offer the catalogue; one grouping, or it is not a
    grouping."""
    from spacr.qt.screens.model_zoo import ModelZooScreen

    page = ModelZooScreen(threaded=False)
    qtbot.addWidget(page)
    assert set(page.sources.enabled()) == {"cellposeSAM", "spaCR"}
    page.set_entries(list(mz.catalogue(remote=True, block=False)))
    shown = set()
    for row, (stem, pairs) in enumerate(page._groups):
        if page._table.isRowHidden(row):
            continue
        shown.add(mz.source_of(pairs[page._chosen[stem]][1]))
    assert shown <= {"cellposeSAM", "spaCR"}
    assert "cellpose3" not in shown
    page.sources.set_on("cellpose3", True)
    folded_in = {mz.source_of(pairs[page._chosen[stem]][1])
                 for row, (stem, pairs) in enumerate(page._groups)
                 if not page._table.isRowHidden(row)}
    assert "cellpose3" in folded_in


def test_the_make_masks_mode_box_offers_only_the_headings_that_are_on(qapp):
    """A user who folded bioimage.io away has said they do not want those
    models; the Mode box must not be the one place that ignores them."""
    from spacr.qt.screens import make_masks as mm

    offered = {mz.source_of(entry) for _k, _p, entry in
               mm._zoo_cellpose_models()}
    assert offered <= set(mzp.remembered_sources())
    assert "bioimage.io" not in offered


def test_the_headings_are_remembered_across_a_reopen(qapp, tmp_path,
                                                     monkeypatch):
    """Like the download folder, and through the same QSettings."""
    monkeypatch.setattr(mzp, "DEFAULT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(tmp_path))
    first = mzp.ModelZooPicker()
    try:
        first.sources.set_on("bioimage.io", True)
        first.sources.set_on("spaCR", False)
    finally:
        first._stop_any_download()
        first.deleteLater()
    second = mzp.ModelZooPicker()
    try:
        assert second.sources.is_on("bioimage.io")
        assert not second.sources.is_on("spaCR")
    finally:
        second._stop_any_download()
        second.deleteLater()
