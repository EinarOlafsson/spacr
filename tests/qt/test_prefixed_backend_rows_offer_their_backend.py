"""Items 551-553, the GUI's half: StarDist, InstanSeg and Omnipose models
are one choice away, and say what they need.

* A model row whose backend is missing reads "needs the <backend> backend",
  its card says so with the backend's licence, Download becomes Install and
  pressing it opens the backend's install (507's dialog); Use stays grey.
* With the backend installed, Use writes ``<prefix><model>``.
* The Live and Timelapse previews route the prefix through the run's own
  table and show the model's probability; the preview's model box accepts
  a model of the backend's own, a bare prefix and a path that exists.
* Make Masks lists every model in its Mode box (grey until installed), an
  installed backend's models in its Model list, and loads them through the
  backend.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import spacr._segmentation_backends as SB
from spacr import model_zoo
from spacr.qt.widgets import model_zoo_picker as mzp

PREFIXED = model_zoo.PREFIXED_KINDS


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """No community fetch, no bioimage.io rows, own backends folder, and
    Show alpha features on: these backends ship as alpha (item 569)."""
    from spacr.qt import preferences

    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    monkeypatch.setattr(SB, "_PROBED", {})
    monkeypatch.setattr(model_zoo, "shared_catalogue_is_stale",
                        lambda *a, **k: False)
    monkeypatch.setattr(model_zoo, "bioimageio_entries", lambda **kw: [])
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)


def test_with_alpha_features_off_every_row_and_mode_is_hidden(
        tmp_path, monkeypatch):
    """Items 551-553 are future-list work: their Model Zoo rows, Mode box
    entries and Model list entries are registered with the alpha gate and
    hidden until Show alpha features is on."""
    from spacr import settings as spacr_settings
    from spacr.qt import preferences
    from spacr.qt.screens import make_masks as mm
    from spacr.qt.screens.model_zoo import _model_is_alpha_hidden

    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: False)
    registered = spacr_settings._alpha_names("models")
    rows = [e for e in model_zoo.installable_backend_entries()
            if e.uri.partition("backend:")[2] in PREFIXED]
    rows += model_zoo._prefixed_model_entries()
    assert rows and all(row.key in registered for row in rows)
    assert all(_model_is_alpha_hidden(row) for row in rows)
    monkeypatch.setattr(mm, "_MAGNIFIER_BACKENDS",
                        dict(mm._MAGNIFIER_BACKENDS))
    monkeypatch.setattr(mm, "_MAGNIFIER_SEGMENTERS",
                        dict(mm._MAGNIFIER_SEGMENTERS))
    assert mm._offer_prefixed_modes() == []
    for name in PREFIXED:
        _install(tmp_path, name)
    assert mm._zoo_prefixed_models() == []
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    assert not any(_model_is_alpha_hidden(row) for row in rows)
    assert mm._offer_prefixed_modes()
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: False)
    mm._offer_prefixed_modes()
    assert not any(SB._prefixed_backend(mode) for mode in mm._MAGNIFIER_BACKENDS)


def _install(tmp_path, name):
    env = tmp_path / "backends" / name
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": name})
    return env


@pytest.fixture
def picker(qapp, tmp_path):
    dialog = mzp.ModelZooPicker(kinds=model_zoo._mask_model_kinds())
    dialog.folder_edit.setText(str(tmp_path / "models"))
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


@pytest.mark.parametrize("name", PREFIXED)
def test_a_missing_backend_is_said_and_install_is_offered(picker, monkeypatch,
                                                         name):
    spec = SB._SPECS[name]
    offered = []
    monkeypatch.setattr(mzp, "install_backend",
                        lambda parent, backend, **kw: offered.append(backend))
    monkeypatch.setattr(picker, "_row_clicked", lambda item: None)
    entry = _select(picker, f"{name}_{spec.default_model}")
    assert mzp._needs_install(entry)
    assert mzp._status_text(entry, entry.path) == (
        f"needs the {spec.label} backend")
    assert picker.download_button.text() == "Install"
    assert not picker.use_button.isEnabled()
    card = picker.card.toPlainText()
    assert f"Needs the {spec.label} backend" in card
    assert "press Install to install it" in card
    assert spec.licence in card
    picker._download_selected()
    assert offered == [name]


@pytest.mark.parametrize("name", PREFIXED)
def test_use_writes_the_prefixed_setting(qapp, tmp_path, name):
    spec = SB._SPECS[name]
    _install(tmp_path, name)
    dialog = mzp.ModelZooPicker(kinds=model_zoo._mask_model_kinds())
    try:
        entry = _select(dialog, f"{name}_{spec.models[-1]}")
        assert not mzp._needs_install(entry)
        assert dialog.use_button.isEnabled()
        card = dialog.card.toPlainText()
        assert f"Runs through the {spec.label} backend, which is installed" \
            in card
        assert f"{spec.prefix}{spec.models[-1]}" in card
        chosen = []
        dialog.model_chosen.connect(chosen.append)
        dialog._accept_selected()
        assert chosen == [f"{spec.prefix}{spec.models[-1]}"]
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("name", PREFIXED)
def test_the_backend_row_says_where_it_is_chosen(name):
    rows = {e.key: e for e in model_zoo.installable_backend_entries()}
    text = mzp._where_a_backend_is_chosen(rows[f"{name}_v1"])
    assert f"{SB._SPECS[name].prefix}<model>" in text
    assert "segmentation_backend" not in text


def test_the_zoo_screen_says_what_a_model_row_needs(tmp_path):
    from spacr.qt.screens.model_zoo import _status_of

    rows = {e.key: e for e in model_zoo._prefixed_model_entries()}
    row = rows["stardist_2D_versatile_fluo"]
    assert _status_of(row) == "needs the StarDist backend"
    _install(tmp_path, "stardist")
    rows = {e.key: e for e in model_zoo._prefixed_model_entries()}
    assert _status_of(rows["stardist_2D_versatile_fluo"]) == "usable"


def test_the_install_path_reaches_the_backend_install(monkeypatch):
    asked = []
    monkeypatch.setattr(mzp, "install_backend",
                        lambda parent, name, **kw: asked.append(name) or True)
    for name in PREFIXED:
        row = model_zoo._prefixed_model_entries()
        entry = next(e for e in row if e.kind == name)
        assert mzp.install_backend_package(None, entry) is True
    assert asked == list(PREFIXED)


# ---------------------------------------------------------------------------
# The previews
# ---------------------------------------------------------------------------

def _preview_backend(monkeypatch):
    from tests.qt import test_cellpose_3_models_are_selectable as cp3

    return cp3._preview_backend(monkeypatch)


@pytest.mark.parametrize("name", PREFIXED)
def test_a_preview_of_a_prefixed_model_runs_in_its_backend(monkeypatch, name):
    LP, loads, calls = _preview_backend(monkeypatch)
    prefix = SB._SPECS[name].prefix
    image = np.random.default_rng(1).random((16, 16, 2)).astype(np.float32)
    request = LP.PreviewRequest(
        image=image, model=f"{prefix}{SB._SPECS[name].default_model}",
        diameter=25.0, flow_threshold=0.6, cellprob=-0.5,
        channels={"cell": 0, "nucleus": 1}, object_types=("nucleus",))
    masks, _flows = LP._segment_multi(request)
    assert [load["name"] for load in loads] == [name]
    assert loads[0]["object_type"] == "nucleus"
    [call] = calls
    assert call["shapes"] == [(16, 16, 2 if name == "instanseg" else 1)]
    assert call["normalize"] is False and call["diameter"] == 25.0
    assert set(np.unique(masks["nucleus"])) == {0, 1, 2}
    assert "nucleus" in request.cellprob_maps


def test_the_preview_model_box_takes_a_prefixed_value(tmp_path):
    from spacr.qt.widgets import live_preview as LP

    folder = tmp_path / "own_model"
    folder.mkdir()
    for name in PREFIXED:
        spec = SB._SPECS[name]
        assert LP._is_a_real_model_name(spec.prefix)
        assert LP._is_a_real_model_name(spec.prefix + spec.models[0])
        assert LP._is_a_real_model_name(f"{spec.prefix}{folder}")
        assert not LP._is_a_real_model_name(spec.prefix + "nope")
        assert not LP._checkpoint_is_missing(spec.prefix + spec.models[0])
        assert not LP._checkpoint_is_missing(f"{spec.prefix}{folder}")
        assert LP._checkpoint_is_missing(f"{spec.prefix}{tmp_path}/gone")


def test_the_timelapse_preview_segments_a_prefixed_frame(monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP

    LP, loads, calls = _preview_backend(monkeypatch)
    monkeypatch.setattr(TP, "preview_cellpose_model",
                        LP.preview_cellpose_model)
    mask = TP.segment_frame(
        np.random.default_rng(2).random((16, 16)).astype(np.float32),
        {"model": "stardist:", "diameter": 18.0})
    assert loads[0]["name"] == "stardist"
    assert calls[0]["shapes"] == [(16, 16, 1)]
    assert mask.dtype == np.int32 and set(np.unique(mask)) == {0, 1, 2}


def test_the_zoo_buttons_ask_for_the_prefixed_kinds(qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.widgets import live_preview as LP
    from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid

    asked = []

    def choose(parent, kinds=None):
        asked.append(kinds)
        return "stardist:2D_versatile_fluo"

    monkeypatch.setattr(mzp, "choose_model", choose)
    host = types.SimpleNamespace(_model_box=QComboBox())
    LP.LivePreviewPanel._choose_a_preview_model(host)
    assert asked == [model_zoo._mask_model_kinds()]
    assert host._model_box.currentText() == "stardist:2D_versatile_fluo"
    stored = []
    grid = types.SimpleNamespace(
        MODEL_KINDS=ObjectSettingsGrid.MODEL_KINDS,
        set_value=lambda question, obj, value: stored.append(value) or True)
    assert ObjectSettingsGrid.choose_model_for(grid, "nucleus")
    assert asked[-1] == model_zoo._mask_model_kinds()
    assert stored == ["stardist:2D_versatile_fluo"]


# ---------------------------------------------------------------------------
# Make Masks
# ---------------------------------------------------------------------------

def test_make_masks_labels_a_prefixed_model_from_the_zoo(qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens import make_masks as mm

    asked = []

    def choose(parent, kinds=None):
        asked.append(kinds)
        return "stardist:2D_versatile_he"

    monkeypatch.setattr(mzp, "choose_model", choose)
    combo = QComboBox()
    host = types.SimpleNamespace(_cp_model=combo,
                                 _fill_zoo_models=lambda: None)
    chosen = mm.MakeMasksScreen._choose_cellpose_model_from_zoo(host)
    assert asked == [model_zoo._mask_model_kinds()]
    assert chosen == "stardist:2D_versatile_he"
    assert combo.currentText() == "StarDist · 2D_versatile_he"


def test_make_masks_loads_a_prefixed_model_in_its_backend(monkeypatch):
    from spacr.qt.screens import make_masks as mm

    loaded = []

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return "backend-model"

    monkeypatch.setattr(SB, "_load_backend", load)
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    monkeypatch.setattr(mm, "_BACKEND_MODEL_ENVS", {})
    assert mm.load_cellpose_model("stardist:2D_versatile_fluo") == \
        "backend-model"
    assert loaded == [("stardist", {"model_name": "2D_versatile_fluo"})]


def test_make_masks_mode_box_offers_every_prefixed_model(tmp_path,
                                                         monkeypatch):
    from spacr.qt.screens import make_masks as mm

    monkeypatch.setattr(mm, "_MAGNIFIER_BACKENDS",
                        dict(mm._MAGNIFIER_BACKENDS))
    monkeypatch.setattr(mm, "_MAGNIFIER_SEGMENTERS",
                        dict(mm._MAGNIFIER_SEGMENTERS))
    added = mm._offer_prefixed_modes()
    expected = [SB._SPECS[name].prefix + model for name in PREFIXED
                for model in SB._SPECS[name].models]
    assert added == expected
    assert mm._offer_prefixed_modes() == []
    mode = "stardist:2D_versatile_fluo"
    assert mm._MAGNIFIER_BACKENDS[mode] == (
        "stardist", "StarDist · 2D_versatile_fluo")
    assert mm._MAGNIFIER_SEGMENTERS[mode] is mm._backend_segmenter
    assert not mm._backend_ready(mode)
    _install(tmp_path, "stardist")
    assert mm._backend_ready(mode)


def test_make_masks_model_list_offers_an_installed_backends_models(
        qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens import make_masks as mm

    monkeypatch.setattr(mm, "_zoo_cellpose_models", lambda: [])
    monkeypatch.setattr(mm, "_zoo_cellpose_dino_models", lambda: [])
    assert mm._zoo_prefixed_models() == []
    _install(tmp_path, "stardist")
    spec = SB._SPECS["stardist"]
    assert mm._zoo_prefixed_models() == [
        (f"stardist:{model}", f"StarDist · {model}") for model in spec.models]
    combo = QComboBox()
    combo.addItem("cpsam", "cpsam")
    host = types.SimpleNamespace(_cp_model=combo, _cp_fetched={})
    mm.MakeMasksScreen._fill_zoo_models(host)
    rows = [combo.itemData(i) for i in range(combo.count())]
    assert rows == ["cpsam"] + [f"stardist:{m}" for m in spec.models]
    assert combo.currentData() == "cpsam"
