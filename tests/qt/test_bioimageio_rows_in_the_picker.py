"""Item 504, the picker's half: bioimage.io rows are offered, used, or
plainly refused.

* A row spaCR cannot run says so in its Status cell, on the status line and
  on its card, and neither Download nor Use is offered.
* A downloaded Cellpose 3 row's Use writes ``cellpose3:<path>``; a
  downloaded Cellpose-SAM row's writes its path.
* The bioimage.io warm-up redraws the table when it brings new rows, rather
  than leaving the category empty until the zoo is opened again.
"""
from __future__ import annotations

import pytest

from spacr import model_zoo
from spacr.qt.widgets import model_zoo_picker as mzp

STANDIN = model_zoo.ModelEntry(
    key="bioimageio_oc1_project_11_cellpose",
    name="oc1_project_11_cellpose.pth", kind="cellpose3",
    source="bioimage.io", uri="", licence="CC-BY-NC-4.0",
    trained_on="OC1 Project 11 Cellpose — Segmentation of Epithelial Cells",
    notes=(model_zoo._CANNOT_RUN + "the weights file bioimage.io publishes "
           "for it is a 1596-byte stand-in, not a Cellpose network.",
           "bioimage.io model happy-elephant"))
CYTO3 = model_zoo.ModelEntry(
    key="bioimageio_cellpose_cyto3", name="cellpose_cyto3.pth",
    kind="cellpose3", source="bioimage.io",
    uri="https://hypha.aicell.io/bioimage-io/artifacts/famous-fish/files/cyto3.pth",
    sha256="2dc3087a", licence="BSD-3-Clause",
    trained_on="CellPose(cyto3) — CellPose 'cyto3' model",
    notes=(f"bioimage.io model famous-fish; {model_zoo._CELLPOSE3_USE}",))
SAM = model_zoo.ModelEntry(
    key="bioimageio_cellpose_sam", name="cellpose_sam", kind="cellpose",
    source="bioimage.io",
    uri="https://hypha.aicell.io/bioimage-io/artifacts/idealistic-eagle/files/cpsam",
    sha256="0f1cc3f7", licence="BSD-3-Clause",
    trained_on="Cellpose-SAM — the generalist",
    notes=(f"bioimage.io model idealistic-eagle; {model_zoo._CELLPOSE4_USE}",))


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No community fetch, and bioimage.io's rows are these three."""
    monkeypatch.setattr(model_zoo, "shared_catalogue_is_stale",
                        lambda *a, **k: False)
    monkeypatch.setattr(model_zoo, "bioimageio_entries",
                        lambda **kwargs: [STANDIN, CYTO3, SAM])


@pytest.fixture
def picker(qapp, tmp_path):
    dialog = mzp.ModelZooPicker(kinds=("cellpose", "cellpose3"))
    dialog.folder_edit.setText(str(tmp_path))
    dialog.sources.set_on("bioimage.io", True)
    dialog.refresh()
    yield dialog
    dialog._stop_any_download()
    dialog.deleteLater()


def _select(dialog, key):
    for stem, pairs in dialog._groups:
        entry = pairs[dialog._chosen[stem]][1]
        if getattr(entry, "key", "") == key:
            dialog.table.selectRow(dialog._row_of_group(
                dialog._groups.index((stem, pairs))))
            return entry
    pytest.fail(f"{key} is not listed")


def test_a_row_spacr_cannot_run_says_so_and_offers_nothing(picker):
    entry = _select(picker, STANDIN.key)
    assert picker.selected_entry() is entry
    assert mzp._status_text(entry, None) == "spaCR cannot run this"
    assert not picker.download_button.isEnabled()
    assert not picker.use_button.isEnabled()
    assert "1596-byte stand-in" in picker.status.text()
    assert "1596-byte stand-in" in picker.card.toPlainText()
    assert "Runs through the Cellpose 3 backend" not in picker.card.toPlainText()


def test_a_runnable_row_offers_its_download_and_says_its_licence(picker):
    _select(picker, CYTO3.key)
    assert picker.download_button.isEnabled()
    assert picker.download_button.text() == "Download"
    assert not picker.use_button.isEnabled(), "nothing is downloaded yet"
    assert "BSD-3-Clause" in picker.card.toPlainText()


@pytest.mark.parametrize("entry, expected", [
    (CYTO3, "cellpose3:{path}"), (SAM, "{path}")])
def test_use_writes_the_setting_that_routes_it(picker, tmp_path, entry,
                                               expected):
    path = tmp_path / entry.name
    path.write_bytes(b"weights")
    picker.refresh()
    _select(picker, entry.key)
    assert picker.use_button.isEnabled()
    chosen = []
    picker.model_chosen.connect(chosen.append)
    picker._accept_selected()
    assert chosen == [expected.format(path=path)]


def test_the_warm_up_redraws_the_table_when_rows_arrive(qapp, qtbot,
                                                        monkeypatch, tmp_path):
    answers = {"network": False}

    def entries(allow_network=False, **kwargs):
        if allow_network:
            answers["network"] = True
        return [CYTO3] if answers["network"] else []

    monkeypatch.setattr(model_zoo, "bioimageio_entries", entries)
    dialog = mzp.ModelZooPicker(kinds=("cellpose", "cellpose3"))
    qtbot.addWidget(dialog)
    try:
        qtbot.waitUntil(lambda: any(
            getattr(pairs[dialog._chosen[stem]][1], "key", "") == CYTO3.key
            for stem, pairs in dialog._groups), timeout=5000)
        assert answers["network"], "the warm-up never asked bioimage.io"
        group = next(
            index for index, (stem, pairs) in enumerate(dialog._groups)
            if getattr(pairs[dialog._chosen[stem]][1], "key", "") == CYTO3.key)
        row = dialog._row_of_group(group)
        assert row is not None, "the new row is in the groups, not the table"
        assert dialog.table.rowCount() == len(dialog._groups)
        assert dialog.table.item(row, 2).text() == CYTO3.trained_on
    finally:
        dialog._stop_any_download()


# ---------------------------------------------------------------------------
# Make Masks
# ---------------------------------------------------------------------------

def test_make_masks_offers_cellpose3_rows_and_lists_what_comes_back(
        qapp, monkeypatch):
    """Its Model zoo… button asks for both Cellposes, and a Cellpose 3
    checkpoint is added to the Model list under the setting that routes it."""
    import types

    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens import make_masks as mm

    asked = []

    def choose(parent, kinds=None):
        asked.append(kinds)
        return "cellpose3:/models/cellpose_cyto3.pth"

    monkeypatch.setattr(mzp, "choose_model", choose)
    combo = QComboBox()
    host = types.SimpleNamespace(_cp_model=combo,
                                 _fill_zoo_models=lambda: None)
    chosen = mm.MakeMasksScreen._choose_cellpose_model_from_zoo(host)
    assert asked == [("cellpose", "cellpose3", "cellpose_dino")]
    assert chosen == "cellpose3:/models/cellpose_cyto3.pth"
    assert combo.currentData() == chosen
    assert combo.currentText() == "Cellpose 3 · cellpose_cyto3.pth"


def test_make_masks_loads_a_cellpose3_model_in_its_backend(monkeypatch):
    """Not Cellpose 4, which would load the checkpoint and segment nonsense:
    the same backend, given the same model, as Mask generation's route."""
    import spacr._segmentation_backends as SB
    from spacr.qt.screens import make_masks as mm

    loaded = []

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return "cellpose3-model"

    monkeypatch.setattr(SB, "_load_backend", load)
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    monkeypatch.setattr(mm, "_BACKEND_MODEL_ENVS", {})
    assert mm.load_cellpose_model(
        "cellpose3:/models/cellpose_cyto3.pth") == "cellpose3-model"
    assert loaded == [("cellpose3",
                       {"model_name": "/models/cellpose_cyto3.pth"})]
