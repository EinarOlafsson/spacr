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
