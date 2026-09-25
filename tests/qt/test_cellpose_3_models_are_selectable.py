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
