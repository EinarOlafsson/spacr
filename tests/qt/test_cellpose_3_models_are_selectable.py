"""Item 503, the Mask panel's half: the four models can be found, chosen,
and bring their own settings with them.

* The zoo's cellpose3 heading is folded away by default, so a user who
  installed the backend saw cyto, cyto2, cyto3 and nuclei nowhere. Once the
  backend is installed the heading comes on, once; turned off afterwards it
  stays off.
* The Cellpose 3 settings rows are hidden until an object's model setting
  names a Cellpose 3 model, and are shown as soon as one does -- including
  when the zoo writes it, which is a ``setText`` and not a commit.
"""
from __future__ import annotations

import pytest

import spacr._segmentation_backends as SB
from spacr.qt.widgets import model_zoo_picker as mzp

LEGACY = ("cellpose3_add_nucleus_channel", "cellpose3_size_model",
          "cellpose3_resample", "cellpose3_augment",
          "cellpose3_percentile_low", "cellpose3_percentile_high")


@pytest.fixture(autouse=True)
def _zoo_preferences_are_put_back(qapp):
    """The headings and the once-only flag are process-wide QSettings."""
    from PySide6.QtCore import QSettings

    keys = (mzp._SOURCES_SETTING, mzp._CELLPOSE3_SHOWN_SETTING)
    settings = QSettings()
    before = {key: settings.value(key) for key in keys
              if settings.contains(key)}
    for key in keys:
        settings.remove(key)
    yield
    settings = QSettings()
    for key in keys:
        settings.remove(key)
        if key in before:
            settings.setValue(key, before[key])


@pytest.fixture
def backend_ready(monkeypatch):
    state = {"ready": True}

    def _state(name, root=None):
        return SB._BackendState(
            name=name, state=SB._INSTALLED if state["ready"] else
            SB._INSTALLABLE, env="/nowhere", in_process=False)

    monkeypatch.setattr(SB, "_backend_state", _state)
    return state


def test_the_heading_comes_on_once_the_backend_is_installed(backend_ready):
    assert "cellpose3" in mzp.remembered_sources()
    assert "cellpose3" in mzp.remembered_sources(), "it did not stay on"
    for name in ("cellposeSAM", "spaCR"):
        assert name in mzp.remembered_sources()


def test_turned_off_afterwards_it_stays_off(backend_ready):
    assert "cellpose3" in mzp.remembered_sources()
    mzp._remember_sources(("cellposeSAM", "spaCR"))
    assert "cellpose3" not in mzp.remembered_sources()


def test_a_users_own_headings_are_kept_when_it_comes_on(backend_ready):
    mzp._remember_sources(("bioimage.io",))
    assert mzp.remembered_sources() == ("bioimage.io", "cellpose3")


def test_without_the_backend_the_heading_stays_as_it_was(backend_ready):
    backend_ready["ready"] = False
    assert "cellpose3" not in mzp.remembered_sources()
    backend_ready["ready"] = True
    assert "cellpose3" in mzp.remembered_sources(), (
        "installing later must still turn it on")


@pytest.mark.parametrize("values, hidden", [
    ({"cell_model_name": "cpsam"}, True),
    ({"cell_model_name": "cyto3"}, True),
    ({"cell_model_name": "cellpose3:cyto3"}, False),
    ({"nucleus_model_name": "cellpose3:nuclei"}, False),
    ({"segmentation_backend": "cellpose3"}, False),
])
def test_the_rows_follow_the_model_settings(values, hidden):
    from spacr.qt.screens.settings_model import keys_hidden_by_their_object

    keys = set(LEGACY) | {"cell_model_name", "nucleus_model_name",
                          "segmentation_backend"}
    settings = {"cell_model_name": "cpsam", "nucleus_model_name": "cpsam",
                "segmentation_backend": "cellpose", **values}
    gone = keys_hidden_by_their_object(keys, settings) & set(LEGACY)
    assert gone == (set(LEGACY) if hidden else set())


def test_a_panel_without_a_model_setting_hides_nothing():
    """The row that would bring them back is not on this panel."""
    from spacr.qt.screens.settings_model import keys_hidden_by_their_object

    assert not keys_hidden_by_their_object(set(LEGACY), {}) & set(LEGACY)


def test_the_mask_panel_shows_them_when_the_zoo_writes_a_cellpose3_model(
        qapp, qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    model = screen._settings_model
    for key in LEGACY:
        assert key in model._widgets, f"the Mask panel has no {key} row"
    model.refresh_object_visibility()
    assert set(LEGACY) <= model._hidden_by_the_run

    field = model._widgets["cell_model_name"]
    field.setText("cellpose3:cyto2")
    qtbot.waitUntil(
        lambda: not (set(LEGACY) & model._hidden_by_the_run), timeout=3000)

    field.setText("cpsam")
    qtbot.waitUntil(
        lambda: set(LEGACY) <= model._hidden_by_the_run, timeout=3000)


# ---------------------------------------------------------------------------
# The Live preview segments a cellpose3: model where the run does (503)
# ---------------------------------------------------------------------------

def _preview_backend(monkeypatch):
    """A stand-in for ``_RemoteBackend`` answering as Cellpose 3 does:
    per-image masks and ``[RGB flow, dP, cell probability, None]``."""
    import types

    import numpy as np

    from spacr.qt.widgets import live_preview as LP

    loads, calls = [], []

    def _eval(x, **kwargs):
        calls.append(dict(kwargs, shapes=[np.shape(i) for i in x]))
        masks, flows = [], []
        for image in x:
            shape = np.shape(image)[:2]
            mask = np.zeros(shape, np.uint16)
            mask[2:8, 2:8] = 1
            mask[10:14, 3:9] = 2
            masks.append(mask)
            flows.append([np.full(shape + (3,), 9, np.uint8),
                          np.zeros((2,) + shape, np.float32),
                          np.full(shape, 1.5, np.float32), None])
        return masks, flows, None

    def _load(name, **kwargs):
        loads.append(dict(kwargs, name=name))
        return types.SimpleNamespace(eval=_eval)

    def _no_cellpose_sam(*args, **kwargs):
        raise AssertionError("the preview built Cellpose-SAM for cellpose3:")

    monkeypatch.setattr(SB, "_load_backend", _load)
    monkeypatch.setattr(LP, "preview_cellpose_model", _no_cellpose_sam)
    return LP, loads, calls


def test_a_preview_of_a_cellpose3_model_runs_in_its_backend(monkeypatch):
    import numpy as np

    LP, loads, calls = _preview_backend(monkeypatch)
    image = np.random.default_rng(1).random((16, 16, 2)).astype(np.float32)
    request = LP.PreviewRequest(
        image=image, model="cellpose3:cyto3", diameter=25.0,
        flow_threshold=0.6, cellprob=-0.5,
        channels={"cell": 0, "nucleus": 1}, object_types=("cell",))
    masks, flows = LP._segment_multi(request)

    assert [load["name"] for load in loads] == ["cellpose3"]
    assert loads[0]["model_name"] == "cellpose3:cyto3"
    [call] = calls
    assert call["shapes"] == [(16, 16, 2)]
    assert call["diameter"] == 25.0
    assert call["flow_threshold"] == 0.6
    assert call["cellprob_threshold"] == -0.5
    assert call["normalize"] == {"normalize": True, "percentile": [1.0, 99.0]}
    assert set(np.unique(masks["cell"])) == {0, 1, 2}
    assert masks["cell"].dtype == np.int32
    assert flows["cell"].shape == (16, 16, 3)
    assert np.all(flows["cell"] == 9)
    np.testing.assert_array_equal(request.cellprob_maps["cell"],
                                  np.full((16, 16), 1.5, np.float32))


def test_a_nucleus_preview_of_a_cellpose3_model_is_one_plane(monkeypatch):
    import numpy as np

    LP, loads, calls = _preview_backend(monkeypatch)
    request = LP.PreviewRequest(
        image=np.ones((16, 16, 2), np.float32), model="cellpose3:nuclei",
        channels={"cell": 0, "nucleus": 1}, object_types=("nucleus",))
    LP._segment_multi(request)
    assert loads[0]["object_type"] == "nucleus"
    assert calls[0]["shapes"] == [(16, 16)]


def test_the_preview_model_box_takes_a_cellpose3_value(tmp_path):
    from spacr.qt.widgets import live_preview as LP

    weights = tmp_path / "cp3_weights"
    weights.write_bytes(b"")
    assert LP._is_a_real_model_name("cellpose3:cyto3")
    assert LP._is_a_real_model_name(f"cellpose3:{weights}")
    assert not LP._is_a_real_model_name("cellpose3:cyto9")
    assert not LP._checkpoint_is_missing("cellpose3:cyto3")
    assert not LP._checkpoint_is_missing(f"cellpose3:{weights}")
    assert LP._checkpoint_is_missing(f"cellpose3:{tmp_path}/gone.pth")


def test_the_timelapse_preview_segments_a_cellpose3_frame_in_its_backend(
        monkeypatch):
    """It shares the model box rules above, so it has to follow them."""
    import numpy as np

    from spacr.qt.widgets import timelapse_preview as TP

    _LP, loads, calls = _preview_backend(monkeypatch)
    monkeypatch.setattr(TP, "preview_cellpose_model",
                        _LP.preview_cellpose_model)
    mask = TP.segment_frame(
        np.random.default_rng(2).random((16, 16)).astype(np.float32),
        {"model": "cellpose3:cyto2", "diameter": 18.0})
    assert loads[0]["model_name"] == "cellpose3:cyto2"
    assert calls[0]["diameter"] == 18.0
    assert mask.dtype == np.int32 and set(np.unique(mask)) == {0, 1, 2}
