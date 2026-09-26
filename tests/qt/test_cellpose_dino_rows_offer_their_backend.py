"""Item 525, the picker's half: a Cellpose-DINO row says what it needs.

* With the Cellpose-DINO backend missing the row's Status reads "needs the
  Cellpose-DINO backend", its card says so and carries the DINOv3 licence,
  Download becomes Install, Use stays grey, and pressing Install opens the
  backend's install -- 507's dialog, with its feedback.
* With the backend installed the row downloads like any other, and Use
  writes ``cellpose_dino:<path>``.
* Make Masks' Model zoo… button asks for Cellpose-DINO rows and lists what
  comes back under its own label; its loader sends the value to the
  Cellpose-DINO backend.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import spacr._segmentation_backends as SB
from spacr import model_zoo
from spacr.qt.widgets import model_zoo_picker as mzp

DINO = model_zoo.ModelEntry(
    key="bioimageio_cellposedino_vit_b_2d_microscopy_instance_segmenter",
    name="cellposedino_vit_b_2d_microscopy_instance_segmenter",
    kind="cellpose_dino", source="bioimage.io",
    uri="https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpdino-vitb",
    sha256="3ed4c06a", licence="BSD-3-Clause",
    trained_on="CellposeDINO ViT-B 2D Microscopy Instance Segmenter",
    notes=(f"bioimage.io model passionate-bug; "
           f"{model_zoo._CELLPOSE_DINO_USE}",))


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """No community fetch, bioimage.io's one row is DINO, own backends."""
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    monkeypatch.setattr(SB, "_PROBED", {})
    monkeypatch.setattr(model_zoo, "shared_catalogue_is_stale",
                        lambda *a, **k: False)
    monkeypatch.setattr(model_zoo, "bioimageio_entries",
                        lambda **kwargs: [DINO])


def _install(tmp_path):
    env = tmp_path / "backends" / "cellpose_dino"
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": "cellpose_dino"})
    return env


@pytest.fixture
def picker(qapp, tmp_path):
    dialog = mzp.ModelZooPicker(
        kinds=("cellpose", "cellpose3", "cellpose_dino"))
    dialog.folder_edit.setText(str(tmp_path / "models"))
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


def test_a_missing_backend_is_said_and_install_is_offered(picker,
                                                         monkeypatch):
    offered = []
    monkeypatch.setattr(mzp, "install_backend",
                        lambda parent, name, **kw: offered.append(name))
    monkeypatch.setattr(picker, "_row_clicked", lambda item: None)
    entry = _select(picker, DINO.key)
    assert mzp._needs_install(entry)
    assert mzp._status_text(entry, None) == "needs the Cellpose-DINO backend"
    assert picker.download_button.text() == "Install"
    assert picker.download_button.isEnabled()
    assert not picker.use_button.isEnabled()
    card = picker.card.toPlainText()
    assert "Needs the Cellpose-DINO backend" in card
    assert "press Install to install it" in card
    assert "DINOv3 License" in card and "NOT open source" in card
    picker._download_selected()
    assert offered == ["cellpose_dino"]


def test_a_downloaded_model_waits_for_its_backend(picker, tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / DINO.name).write_bytes(b"weights")
    picker.refresh()
    _select(picker, DINO.key)
    assert not picker.use_button.isEnabled(), (
        "a DINO model is not usable until its backend is installed")
    assert picker.download_button.text() == "Install"


def test_an_installed_backend_makes_the_row_an_ordinary_download(picker,
                                                                 tmp_path):
    _install(tmp_path)
    picker.refresh()
    entry = _select(picker, DINO.key)
    assert not mzp._needs_install(entry)
    assert mzp._status_text(entry, None) == "not downloaded"
    assert picker.download_button.text() == "Download"
    assert picker.download_button.isEnabled()
    card = picker.card.toPlainText()
    assert "Runs through the Cellpose-DINO backend, which is installed" in card


def test_use_writes_the_cellpose_dino_setting(picker, tmp_path):
    _install(tmp_path)
    folder = tmp_path / "models"
    folder.mkdir()
    path = folder / DINO.name
    path.write_bytes(b"weights")
    picker.refresh()
    entry = _select(picker, DINO.key)
    assert mzp._status_text(entry, str(path)) == "on this machine"
    assert picker.use_button.isEnabled()
    chosen = []
    picker.model_chosen.connect(chosen.append)
    picker._accept_selected()
    assert chosen == [f"cellpose_dino:{path}"]


def test_the_backend_row_says_where_it_is_chosen(tmp_path):
    rows = {e.key: e for e in model_zoo.installable_backend_entries()}
    text = mzp._where_a_backend_is_chosen(rows["cellpose_dino_v1"])
    assert "Cellpose-DINO model from the bioimage.io heading" in text
    assert "segmentation_backend" not in text


def test_the_install_button_path_reaches_the_backend_install(qapp,
                                                             monkeypatch):
    asked = []
    monkeypatch.setattr(mzp, "install_backend",
                        lambda parent, name, **kw: asked.append(name) or True)
    assert mzp.install_backend_package(None, DINO) is True
    assert asked == ["cellpose_dino"]


def test_make_masks_asks_for_dino_rows_and_labels_them(qapp, monkeypatch):
    import types

    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens import make_masks as mm

    asked = []

    def choose(parent, kinds=None):
        asked.append(kinds)
        return "cellpose_dino:/models/cellposedino_vit_b"

    monkeypatch.setattr(mzp, "choose_model", choose)
    combo = QComboBox()
    host = types.SimpleNamespace(_cp_model=combo,
                                 _fill_zoo_models=lambda: None)
    chosen = mm.MakeMasksScreen._choose_cellpose_model_from_zoo(host)
    assert asked == [("cellpose", "cellpose3", "cellpose_dino")]
    assert chosen == "cellpose_dino:/models/cellposedino_vit_b"
    assert combo.currentData() == chosen
    assert combo.currentText() == "Cellpose-DINO · cellposedino_vit_b"


def test_make_masks_loads_a_dino_model_in_its_backend(monkeypatch):
    from spacr.qt.screens import make_masks as mm

    loaded = []

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return "dino-model"

    monkeypatch.setattr(SB, "_load_backend", load)
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    monkeypatch.setattr(mm, "_BACKEND_MODEL_ENVS", {})
    assert mm.load_cellpose_model(
        "cellpose_dino:/models/cellposedino_vit_b") == "dino-model"
    assert loaded == [("cellpose_dino",
                       {"model_name": "/models/cellposedino_vit_b"})]


def test_mask_generation_model_fields_offer_dino_rows():
    from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid

    assert "cellpose_dino" in ObjectSettingsGrid.MODEL_KINDS


# ---------------------------------------------------------------------------
# The previews and Make Masks' boxes run cellpose_dino: in its backend (525)
# ---------------------------------------------------------------------------

def _preview_backend(monkeypatch):
    from tests.qt import test_cellpose_3_models_are_selectable as cp3

    return cp3._preview_backend(monkeypatch)


def test_a_preview_of_a_dino_model_runs_in_its_backend(monkeypatch):
    """Routed by the run's table, given the SAM call, and its RGB flow and
    cell probability land in the Flows and Cell probability views."""
    import numpy as np

    LP, loads, calls = _preview_backend(monkeypatch)
    image = np.random.default_rng(1).random((16, 16, 2)).astype(np.float32)
    request = LP.PreviewRequest(
        image=image, model="cellpose_dino:/m/cellposedino_vit_b",
        diameter=25.0, flow_threshold=0.6, cellprob=-0.5,
        channels={"cell": 0, "nucleus": 1}, object_types=("cell",))
    masks, flows = LP._segment_multi(request)

    assert [load["name"] for load in loads] == ["cellpose_dino"]
    assert loads[0]["model_name"] == "cellpose_dino:/m/cellposedino_vit_b"
    [call] = calls
    assert call["shapes"] == [(16, 16, 2)]
    assert call["normalize"] is False and call["channel_axis"] == -1
    assert call["diameter"] == 25.0
    assert (call["flow_threshold"], call["cellprob_threshold"]) == (0.6, -0.5)
    assert masks["cell"].dtype == np.int32
    assert set(np.unique(masks["cell"])) == {0, 1, 2}
    assert flows["cell"].shape == (16, 16, 3) and np.all(flows["cell"] == 9)
    np.testing.assert_array_equal(request.cellprob_maps["cell"],
                                  np.full((16, 16), 1.5, np.float32))


def test_a_nucleus_preview_of_a_dino_model_is_one_plane(monkeypatch):
    import numpy as np

    LP, loads, calls = _preview_backend(monkeypatch)
    LP._segment_multi(LP.PreviewRequest(
        image=np.ones((16, 16, 2), np.float32), model="cellpose_dino:/m/w",
        channels={"cell": 0, "nucleus": 1}, object_types=("nucleus",)))
    assert loads[0]["object_type"] == "nucleus"
    assert calls[0]["shapes"] == [(16, 16, 1)]


def test_the_preview_model_box_takes_a_dino_value(tmp_path):
    from spacr.qt.widgets import live_preview as LP

    weights = tmp_path / "cellposedino_vit_b"
    weights.write_bytes(b"")
    assert LP._is_a_real_model_name(f"cellpose_dino:{weights}")
    assert not LP._is_a_real_model_name("cellpose_dino:")
    assert not LP._is_a_real_model_name(f"cellpose_dino:{tmp_path}/gone")
    assert not LP._checkpoint_is_missing(f"cellpose_dino:{weights}")
    assert LP._checkpoint_is_missing(f"cellpose_dino:{tmp_path}/gone")
    assert LP._checkpoint_is_missing("cellpose_dino:")


def test_the_timelapse_preview_segments_a_dino_frame_in_its_backend(
        monkeypatch):
    import numpy as np

    from spacr.qt.widgets import timelapse_preview as TP

    LP, loads, calls = _preview_backend(monkeypatch)
    monkeypatch.setattr(TP, "preview_cellpose_model",
                        LP.preview_cellpose_model)
    mask = TP.segment_frame(
        np.random.default_rng(2).random((16, 16)).astype(np.float32),
        {"model": "cellpose_dino:/m/w", "diameter": 18.0})
    assert loads[0]["name"] == "cellpose_dino"
    assert calls[0]["diameter"] == 18.0
    assert calls[0]["shapes"] == [(16, 16, 1)]
    assert mask.dtype == np.int32 and set(np.unique(mask)) == {0, 1, 2}


def test_the_preview_zoo_button_asks_for_dino_rows(qapp, monkeypatch):
    import types

    from PySide6.QtWidgets import QComboBox

    from spacr.qt.widgets import live_preview as LP

    asked = []

    def choose(parent, kinds=None):
        asked.append(kinds)
        return "cellpose_dino:/m/w"

    monkeypatch.setattr(mzp, "choose_model", choose)
    host = types.SimpleNamespace(_model_box=QComboBox())
    LP.LivePreviewPanel._choose_a_preview_model(host)
    assert asked == [("cellpose", "cellpose3", "cellpose_dino")]
    assert host._model_box.currentText() == "cellpose_dino:/m/w"


def _downloaded_dino(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / DINO.name).write_bytes(b"weights")
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(folder))
    monkeypatch.setattr(mzp, "remembered_sources",
                        lambda: tuple(model_zoo.ZOO_SOURCES))
    monkeypatch.setattr(model_zoo, "catalogue", lambda **kwargs: [DINO])
    return f"cellpose_dino:{folder / DINO.name}"


def test_make_masks_lists_downloaded_dino_models(tmp_path, monkeypatch):
    from spacr.qt.screens import make_masks as mm

    value = _downloaded_dino(tmp_path, monkeypatch)
    assert mm._zoo_cellpose_dino_models() == [
        (value, f"Cellpose-DINO · {DINO.name}")]
    assert mm._zoo_cellpose_models() == []
    (tmp_path / "models" / DINO.name).unlink()
    assert mm._zoo_cellpose_dino_models() == []


def test_make_masks_model_list_offers_downloaded_dino_models(qapp, tmp_path,
                                                             monkeypatch):
    import types

    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens import make_masks as mm

    value = _downloaded_dino(tmp_path, monkeypatch)
    combo = QComboBox()
    combo.addItem("cpsam", "cpsam")
    host = types.SimpleNamespace(_cp_model=combo, _cp_fetched={})
    mm.MakeMasksScreen._fill_zoo_models(host)
    mm.MakeMasksScreen._fill_zoo_models(host)
    rows = [(combo.itemText(i), combo.itemData(i))
            for i in range(combo.count())]
    assert rows == [("cpsam", "cpsam"),
                    (f"Cellpose-DINO · {DINO.name}", value)]
    assert combo.currentData() == "cpsam"


def test_make_masks_mode_box_offers_downloaded_dino_models(tmp_path,
                                                           monkeypatch):
    """Each downloaded checkpoint becomes a magnifier mode of the
    Cellpose-DINO backend, segmented through ``_backend_model``."""
    from spacr.qt.screens import make_masks as mm

    value = _downloaded_dino(tmp_path, monkeypatch)
    monkeypatch.setattr(mm, "_MAGNIFIER_BACKENDS",
                        dict(mm._MAGNIFIER_BACKENDS))
    monkeypatch.setattr(mm, "_MAGNIFIER_SEGMENTERS",
                        dict(mm._MAGNIFIER_SEGMENTERS))
    assert mm._offer_cellpose_dino_modes() == [value]
    assert mm._offer_cellpose_dino_modes() == []
    assert mm._MAGNIFIER_BACKENDS[value] == (
        "cellpose_dino", f"Cellpose-DINO · {DINO.name}")
    assert mm._MAGNIFIER_SEGMENTERS[value] is mm._backend_segmenter
    assert not mm._backend_ready(value)
    _install(tmp_path)
    assert mm._backend_ready(value)
    assert mm._magnifier_mode_label(value) == f"Cellpose-DINO · {DINO.name}"

    loaded = []

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return "dino-model"

    monkeypatch.setattr(SB, "_load_backend", load)
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    monkeypatch.setattr(mm, "_BACKEND_MODEL_ENVS", {})
    assert mm._backend_model(value) == "dino-model"
    assert loaded == [("cellpose_dino", {
        "model_name": str(tmp_path / "models" / DINO.name)})]
